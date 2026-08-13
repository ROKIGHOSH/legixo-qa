"""Thin wrapper around the real Pinecone service (not an in-memory fake)."""
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

from app import config


def get_client() -> Pinecone:
    if not config.PINECONE_API_KEY or "dummy" in config.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Copy .env.example to .env and fill in "
            "a real key from https://app.pinecone.io"
        )
    return Pinecone(api_key=config.PINECONE_API_KEY)


def ensure_index():
    """Create the Pinecone index if it doesn't already exist. Idempotent."""
    pc = get_client()
    existing = [idx["name"] for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    return pc.Index(config.PINECONE_INDEX_NAME)


def get_index():
    pc = get_client()
    return pc.Index(config.PINECONE_INDEX_NAME)


def clear_namespace(namespace: str = None):
    """Delete everything in a namespace. Called at the start of ingest so that
    re-running ingest never creates duplicate/stale vectors (see README for why)."""
    namespace = namespace or config.PINECONE_NAMESPACE
    index = ensure_index()
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        # Namespace may not exist yet on a fresh index - that's fine.
        pass


def upsert_chunks(records: List[Dict[str, Any]], namespace: str = None):
    """records: list of {"id", "values", "metadata"} dicts."""
    namespace = namespace or config.PINECONE_NAMESPACE
    index = ensure_index()
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)


def query(vector: List[float], top_k: int, namespace: str = None):
    namespace = namespace or config.PINECONE_NAMESPACE
    index = get_index()
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    return result.get("matches", [])
