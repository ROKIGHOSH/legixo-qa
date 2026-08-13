"""Central place that reads all settings from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legixo-qa")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "legixo-default")

# Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Embeddings (local, free)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = _int("EMBEDDING_DIM", 384)

# Corpus / chunking
CORPUS_DIR = os.getenv("CORPUS_DIR", "gen_ai_takehome_sample_corpus")
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

# Retrieval / graph
TOP_K = _int("TOP_K", 5)
MIN_GOOD_CHUNK_SCORE = _float("MIN_GOOD_CHUNK_SCORE", 0.35)
MAX_GRAPH_STEPS = _int("MAX_GRAPH_STEPS", 6)

# API server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8000)
