FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# GEMINI_API_KEY is required at build time to embed the corpus during ingest.
# Declared as ARG so Render passes the dashboard env var into the build (declare
# the ARG, set the env var in Render's settings). It is set only in this builder
# stage, which is discarded — the runtime image below never receives it.
# Local build:  docker build --build-arg GEMINI_API_KEY=<key> -t smart-grid-rag .
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
RUN uv run ingest.py

RUN uv run python - <<'PY'
from flashrank import Ranker

Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/app/.flashrank")
PY


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/db /app/db
COPY --from=builder /app/.flashrank /app/.flashrank
COPY --from=builder /app/papers /app/papers
COPY --from=builder /app/rag.py /app/rag.py
COPY --from=builder /app/server.py /app/server.py
COPY --from=builder /app/chat.py /app/chat.py
COPY --from=builder /app/ingest.py /app/ingest.py
COPY --from=builder /app/main.py /app/main.py

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port \"${PORT:-8000}\""]