"""HTTP API for the Legixo Q&A take-home.

Run with:  uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.graph import ask as run_graph
from app.ingest import run_ingest
from app import config

app = FastAPI(
    title="Legixo Q&A API",
    description="Small RAG Q&A API over a fictional legal-notes corpus (LangGraph + Pinecone).",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, example="What is the notice period for Priya Nambiar?")
    include_trace: bool = Field(default=False, description="Include the LangGraph step trace in the response.")


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    trace: Optional[List[str]] = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    namespace: str


@app.get("/")
def root():
    return {
        "service": "legixo-qa",
        "endpoints": ["POST /ask", "POST /ingest", "GET /health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Loads the corpus, chunks + embeds it, and upserts into Pinecone.
    Safe to call more than once - it clears the namespace first each time
    so re-running never creates duplicate chunks."""
    try:
        result = run_ingest()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        state = run_graph(payload.question)
    except RuntimeError as e:
        # Missing API keys etc - surface as a clear 400 rather than a stack trace.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = {
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
    }
    if payload.include_trace:
        response["trace"] = state.get("trace", [])
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
