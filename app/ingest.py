"""Ingest pipeline: read corpus files -> chunk -> embed -> upsert to Pinecone.

Re-run behaviour (see README "Pinecone checklist"):
  We DELETE the whole namespace at the start of every ingest run, then
  re-upsert everything with deterministic ids ("<filename>::chunk<N>").
  This means running ingest twice is safe and idempotent - you always end
  up with exactly one copy of each chunk, never duplicates.
"""
import os
import sys
import time
from pathlib import Path

from app import config
from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.vector_store import clear_namespace, upsert_chunks, ensure_index


def load_corpus_files(corpus_dir: str):
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files = sorted(
        [
            p
            for p in corpus_path.iterdir()
            if p.is_file() and p.suffix.lower() in (".md", ".txt")
        ]
    )
    if not files:
        raise FileNotFoundError(f"No .md/.txt files found in {corpus_dir}")
    return files


def run_ingest(corpus_dir: str = None, namespace: str = None):
    corpus_dir = corpus_dir or config.CORPUS_DIR
    namespace = namespace or config.PINECONE_NAMESPACE

    print(f"[ingest] Ensuring Pinecone index '{config.PINECONE_INDEX_NAME}' exists...")
    ensure_index()

    print(f"[ingest] Clearing namespace '{namespace}' (safe re-run: no duplicate chunks)...")
    clear_namespace(namespace)

    files = load_corpus_files(corpus_dir)
    print(f"[ingest] Found {len(files)} source file(s) in {corpus_dir}")

    all_records = []
    all_texts = []
    all_meta = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path.name}::chunk{i}"
            all_texts.append(chunk)
            all_meta.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": file_path.name,
                    "chunk_index": i,
                    "text": chunk,
                }
            )
        print(f"  - {file_path.name}: {len(chunks)} chunk(s)")

    print(f"[ingest] Embedding {len(all_texts)} chunk(s) with '{config.EMBEDDING_MODEL}' "
          f"(local, free, first run downloads the model)...")
    vectors = embed_texts(all_texts)

    for meta, vector in zip(all_meta, vectors):
        all_records.append(
            {
                "id": meta["chunk_id"],
                "values": vector,
                "metadata": meta,
            }
        )

    print(f"[ingest] Upserting {len(all_records)} vector(s) into namespace '{namespace}'...")
    upsert_chunks(all_records, namespace=namespace)

    print("[ingest] Done.")
    return {"files": len(files), "chunks": len(all_records), "namespace": namespace}


if __name__ == "__main__":
    result = run_ingest()
    print(result)
