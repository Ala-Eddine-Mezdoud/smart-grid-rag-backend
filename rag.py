"""Shared RAG engine over the persistent Chroma database built by ingest.py.

Both the terminal client (chat.py) and the FastAPI server (server.py) use this so
that retrieval, the grounding prompt, and answer generation stay identical.

Retrieval uses Google embeddings (via langchain) + BM25 + a cross-encoder reranker.
Generation uses the google-genai Interactions API (the legacy generateContent endpoint
has been retired for newer API projects).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from google import genai
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DB_DIR = Path(__file__).parent / "db"
PAPERS_DIR = Path(__file__).parent / "papers"
COLLECTION_NAME = "smart_grid_papers"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"

# Hybrid retrieval: dense (Chroma) + sparse (BM25) candidates are fused with
# reciprocal rank fusion, then a cross-encoder reranker keeps the best TOP_K.
TOP_K = 10
DENSE_K = 20
BM25_K = 20
POOL_K = 30
RRF_K = 60
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
RERANK_CACHE_DIR = Path(__file__).parent / ".flashrank"

# History-aware query rewriting: how much recent conversation to feed the condenser.
MAX_HISTORY_MESSAGES = 6
HISTORY_CHAR_LIMIT = 600

UNAVAILABLE_MESSAGE = "The information is not available in the provided documents."

SYSTEM_INSTRUCTION = (
    "You are a research assistant answering questions about a corpus of smart grid "
    "research papers.\n"
    "Answer ONLY using the numbered context passages provided in the user's message. Do "
    "not rely on any outside knowledge. If the context does not contain the answer, reply "
    f'exactly: "{UNAVAILABLE_MESSAGE}"\n'
    "Cite every claim with the bracketed number(s) of the passage(s) that support it, e.g. "
    "[1] or [2][3], placed immediately after the claim. Use ONLY these bracket number "
    "markers for citations — never write file names or page numbers in your prose. Use "
    "markdown (headings, lists, bold) when it improves readability."
)

CONDENSE_SYSTEM = (
    "You rewrite a user's follow-up question into a standalone question that can be "
    "understood without the prior conversation. Resolve pronouns and implicit references "
    '("it", "that", "the second one") using the history, and keep the user\'s intent and '
    "terminology. If the question is already standalone, return it unchanged. Return ONLY "
    "the rewritten question, with no preamble or quotes."
)


@dataclass
class Source:
    """A de-duplicated citation pointing at one page of one paper."""

    id: str
    title: str
    page: int
    filename: str
    snippet: str
    index: int = 0  # 1-based citation number the model cites as [index]

    def to_dict(self, base_url: str = "") -> dict:
        url = f"{base_url}papers/{self.filename}#page={self.page}" if base_url else ""
        image = (
            f"{base_url}api/page-image?file={self.filename}&page={self.page}"
            if base_url
            else ""
        )
        return {
            "index": self.index,
            "id": self.id,
            "title": self.title,
            "page": self.page,
            "filename": self.filename,
            "snippet": self.snippet,
            "url": url,
            "image": image,
        }


def _source_from_doc(doc: Document) -> Source:
    filename = Path(doc.metadata.get("source", "unknown")).name
    # PyMuPDF pages are 0-indexed; present them 1-indexed to humans.
    page = int(doc.metadata.get("page", 0)) + 1
    snippet = " ".join(doc.page_content.split())[:240]
    return Source(
        id=f"{filename}#p{page}",
        title=filename,
        page=page,
        filename=filename,
        snippet=snippet,
    )


def _citation_order(docs: list[Document]) -> dict[str, int]:
    """Map each unique (file, page) citation id to its 1-based number, in doc order."""
    order: dict[str, int] = {}
    for doc in docs:
        key = _source_from_doc(doc).id
        if key not in order:
            order[key] = len(order) + 1
    return order


def dedupe_sources(docs: list[Document]) -> list[Source]:
    """Collapse retrieved chunks into unique (file, page) citations, numbered in order."""
    order = _citation_order(docs)
    seen: set[str] = set()
    sources: list[Source] = []
    for doc in docs:
        source = _source_from_doc(doc)
        if source.id in seen:
            continue
        seen.add(source.id)
        source.index = order[source.id]
        sources.append(source)
    return sources


def format_context(docs: list[Document]) -> str:
    """Number the passages so the model can cite them as [1], [2], … consistently."""
    order = _citation_order(docs)
    parts = []
    for doc in docs:
        source = _source_from_doc(doc)
        parts.append(
            f"[{order[source.id]}] {source.title}, page {source.page}:\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def _build_input(question: str, docs: list[Document]) -> str:
    return f"Context:\n{format_context(docs)}\n\nQuestion: {question}"


class RagEngine:
    """Loads the vector store and generation client once and answers against them."""

    def __init__(self, top_k: int = TOP_K) -> None:
        load_dotenv()

        if not DB_DIR.exists():
            raise RuntimeError(
                f"No database found at {DB_DIR}. Run `uv run ingest.py` first."
            )

        self.top_k = top_k
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(DB_DIR),
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": DENSE_K})

        # Build the BM25 sparse index from the exact same chunks stored in Chroma,
        # so lexical matches (acronyms, standard numbers) stay aligned with the dense index.
        stored = self.vectorstore.get(include=["documents", "metadatas"])
        self.bm25_retriever = BM25Retriever.from_texts(
            stored["documents"], metadatas=stored["metadatas"]
        )
        self.bm25_retriever.k = BM25_K

        try:
            self.reranker = Ranker(
                model_name=RERANK_MODEL, cache_dir=str(RERANK_CACHE_DIR)
            )
        except Exception:
            self.reranker = None

        # Generation via the Interactions API (reads GEMINI_API_KEY from the environment).
        self.client = genai.Client()

    def condense_question(
        self, question: str, history: list[dict] | None = None
    ) -> str:
        """Rewrite a follow-up into a standalone question using recent chat history.

        Returns the question unchanged when there is no history (e.g. the first turn).
        """
        if not history:
            return question

        recent = history[-MAX_HISTORY_MESSAGES:]
        lines = []
        for turn in recent:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            speaker = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{speaker}: {content[:HISTORY_CHAR_LIMIT]}")
        if not lines:
            return question

        result = self.client.interactions.create(
            model=CHAT_MODEL,
            input=f"Conversation so far:\n{chr(10).join(lines)}\n\nFollow-up question: {question}",
            system_instruction=CONDENSE_SYSTEM,
        )
        rewritten = (result.output_text or "").strip()
        return rewritten or question

    def retrieve(self, question: str) -> list[Document]:
        dense = self.dense_retriever.invoke(question)
        sparse = self.bm25_retriever.invoke(question)
        fused = self._reciprocal_rank_fusion([dense, sparse])[:POOL_K]
        return self.rerank(question, fused)

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[Document]], k: int = RRF_K
    ) -> list[Document]:
        """Merge ranked result lists by reciprocal rank fusion, deduped on chunk text."""
        scores: dict[str, float] = {}
        docs_by_key: dict[str, Document] = {}
        for docs in ranked_lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                docs_by_key.setdefault(key, doc)
        ordered = sorted(scores, key=scores.get, reverse=True)
        return [docs_by_key[key] for key in ordered]

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        """Reorder dense candidates with the cross-encoder and keep the top_k."""
        if not docs:
            return []
        if self.reranker is None:
            return docs[: self.top_k]
        request = RerankRequest(
            query=question,
            passages=[{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)],
        )
        try:
            ranked = self.reranker.rerank(request)[: self.top_k]
        except Exception:
            return docs[: self.top_k]
        top_docs: list[Document] = []
        for item in ranked:
            doc = docs[int(item["id"])]
            doc.metadata["rerank_score"] = float(item["score"])
            top_docs.append(doc)
        return top_docs

    @staticmethod
    def _text_deltas(stream) -> Iterator[str]:
        """Yield only the answer text deltas from an Interactions stream (skip thoughts)."""
        for event in stream:
            delta = getattr(event, "delta", None)
            if delta is not None and getattr(delta, "type", None) == "text" and delta.text:
                yield delta.text

    def stream(self, question: str, docs: list[Document]) -> Iterator[str]:
        """Yield answer text chunks grounded in the given documents."""
        stream = self.client.interactions.create(
            model=CHAT_MODEL,
            input=_build_input(question, docs),
            system_instruction=SYSTEM_INSTRUCTION,
            stream=True,
        )
        yield from self._text_deltas(stream)

    def answer(self, question: str, docs: list[Document]) -> str:
        result = self.client.interactions.create(
            model=CHAT_MODEL,
            input=_build_input(question, docs),
            system_instruction=SYSTEM_INSTRUCTION,
        )
        return result.output_text
