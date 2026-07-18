"""FastAPI server exposing the smart grid RAG engine over HTTP with streaming.

Run with:  uv run uvicorn server:app --reload
Endpoints:
  GET  /api/health          -> readiness probe
  GET  /api/papers          -> list indexed PDFs
  POST /api/chat            -> Server-Sent Events stream (sources + answer tokens)
  GET  /api/page-image      -> render one PDF page to PNG (citation hover previews)
  GET  /papers/{file}       -> the source PDFs (so citations are clickable)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import pymupdf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from rag import DB_DIR, PAPERS_DIR, UNAVAILABLE_MESSAGE, RagEngine, dedupe_sources

def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [value.strip() for value in raw.split(",") if value.strip()]


# Allow local frontend development by default; production should override this env var.
ALLOWED_ORIGINS = _csv_env(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
)
PORT = int(os.getenv("PORT", "8000"))

engine: RagEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    # Build the vector store + LLM once at startup, not per-request.
    try:
        engine = RagEngine()
    except Exception as exc:  # noqa: BLE001 - fail startup with a clear message
        raise RuntimeError(
            f"Failed to initialize the RAG engine. Make sure the Chroma database exists at {DB_DIR} and GEMINI_API_KEY is set."
        ) from exc

    try:
        yield
    finally:
        engine = None


app = FastAPI(title="Smart Grid RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

if PAPERS_DIR.exists():
    app.mount("/papers", StaticFiles(directory=str(PAPERS_DIR)), name="papers")


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


def _sse(event: str, data: object) -> str:
    """Format a single Server-Sent Event with a JSON-encoded (single-line) payload."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "ready": engine is not None}


@app.get("/api/papers")
async def papers(request: Request) -> dict:
    base_url = str(request.base_url)
    files = sorted(p.name for p in PAPERS_DIR.glob("*.pdf"))
    return {
        "count": len(files),
        "papers": [{"filename": f, "url": f"{base_url}papers/{f}"} for f in files],
    }


@app.get("/api/page-image")
def page_image(file: str, page: int = Query(ge=1)) -> Response:
    """Render one page of a source PDF to a PNG, for citation hover previews."""
    # Path.name strips any directory components, blocking path traversal.
    pdf_path = PAPERS_DIR / Path(file).name
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")

    with pymupdf.open(str(pdf_path)) as doc:
        if page > doc.page_count:
            raise HTTPException(status_code=404, detail="Page out of range")
        pixmap = doc.load_page(page - 1).get_pixmap(dpi=150)
        png = pixmap.tobytes("png")

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    assert engine is not None, "RAG engine is not initialised"
    base_url = str(request.base_url)
    question = body.question.strip()
    history = [turn.model_dump() for turn in body.history]

    def event_stream() -> Iterator[str]:
        try:
            # Rewrite follow-ups into a standalone question so retrieval isn't
            # confused by pronouns/references, then answer the standalone query.
            standalone = engine.condense_question(question, history)
            docs = engine.retrieve(standalone)
            sources = dedupe_sources(docs)
            yield _sse("sources", [s.to_dict(base_url) for s in sources])

            produced = False
            for token in engine.stream(standalone, docs):
                produced = True
                yield _sse("token", {"text": token})

            if not produced:
                yield _sse("token", {"text": UNAVAILABLE_MESSAGE})

            yield _sse("done", {})
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            yield _sse("error", {"message": str(exc)})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
