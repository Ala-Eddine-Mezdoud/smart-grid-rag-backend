"""Ingest all PDFs from ./papers into a persistent Chroma database at ./db."""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAPERS_DIR = Path("papers")
DB_DIR = Path("db")
COLLECTION_NAME = "smart_grid_papers"
EMBEDDING_MODEL = "models/gemini-embedding-001"


def main():
    load_dotenv()

    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        sys.exit(f"No PDF files found in {PAPERS_DIR.resolve()}")

    print(f"Found {len(pdf_paths)} PDF(s) in {PAPERS_DIR}/")

    documents = []
    for pdf_path in pdf_paths:
        print(f"  Loading {pdf_path.name} ...", end=" ", flush=True)
        pages = PyMuPDFLoader(str(pdf_path)).load()
        documents.extend(pages)
        print(f"{len(pages)} pages")

    print(f"Loaded {len(documents)} pages total")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    if DB_DIR.exists():
        print(f"Existing database found at {DB_DIR}/ — deleting it")
        shutil.rmtree(DB_DIR)

    print("Embedding chunks and building the Chroma database (this may take a while) ...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
    )

    print()
    print("Ingestion complete:")
    print(f"  PDFs loaded:    {len(pdf_paths)}")
    print(f"  Pages:          {len(documents)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Database:       {DB_DIR.resolve()}")


if __name__ == "__main__":
    main()
