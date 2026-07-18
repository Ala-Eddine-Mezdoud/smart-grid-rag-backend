---
title: Smart Grid RAG Backend
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Smart Grid RAG Backend

FastAPI service that answers questions about a corpus of smart grid research papers,
grounded strictly in the source PDFs with page-level citations.

- **Retrieval**: hybrid dense (`gemini-embedding-001` + Chroma) + BM25, fused with
  reciprocal rank fusion, reranked by a `flashrank` cross-encoder.
- **Generation**: Google `gemini-3.5-flash` via the GenAI Interactions API, streamed.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Readiness probe |
| `GET` | `/api/papers` | List indexed PDFs |
| `POST` | `/api/chat` | `{ "question": "…", "history": [...] }` → SSE stream (`sources`, `token`, `done`) |
| `GET` | `/api/page-image?file=…&page=…` | Render a PDF page to PNG (citation previews) |
| `GET` | `/papers/{file}` | The source PDFs |

## Configuration

Set these as **Space secrets / variables**:

- `GEMINI_API_KEY` (**required**) — used at build time to embed the corpus and at
  runtime for query embedding + generation.
- `ALLOWED_ORIGINS` (optional) — comma-separated frontend origins for CORS
  (default `http://localhost:3000`).

The Chroma database is built during the Docker build via `ingest.py`.
