"""Interactive terminal chat over the Chroma database built by ingest.py.

This is a thin CLI around the shared engine in rag.py (the same engine the FastAPI
server uses), so terminal and web answers stay identical.
"""

import sys

from rag import RagEngine, dedupe_sources


def main():
    try:
        engine = RagEngine()
    except RuntimeError as exc:
        sys.exit(str(exc))

    print("Smart Grid RAG chat — ask a question, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        docs = engine.retrieve(question)

        print("\nAssistant: ", end="", flush=True)
        for token in engine.stream(question, docs):
            print(token, end="", flush=True)
        print()

        sources = ", ".join(f"{s.title} (p. {s.page})" for s in dedupe_sources(docs))
        print(f"Sources: {sources}\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
