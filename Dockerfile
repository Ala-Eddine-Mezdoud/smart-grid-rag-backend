FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# The Chroma DB is prebuilt and copied in with the source (see COPY above), so the
# build needs no API key. Just pre-download the reranker model so the runtime needs
# no network for it. Single-line RUN — Cloud Build's classic Docker builder does not
# support Dockerfile heredocs.
RUN uv run python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/.flashrank')"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Run as a non-root user (good practice; Chroma/flashrank can write their working files).
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --from=builder --chown=user:user /app/.venv /app/.venv
COPY --from=builder --chown=user:user /app/db /app/db
COPY --from=builder --chown=user:user /app/.flashrank /app/.flashrank
COPY --from=builder --chown=user:user /app/papers /app/papers
COPY --from=builder --chown=user:user \
    /app/rag.py /app/server.py /app/chat.py /app/ingest.py /app/main.py /app/

USER user

EXPOSE 8000

# Cloud Run/Railway inject $PORT; bind it, else default to 8000.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
