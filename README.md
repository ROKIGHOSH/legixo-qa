# Legixo Q&A — Gen AI intern take-home

A small Q&A HTTP API over a fictional legal-notes corpus. Built with **Python + LangGraph + Pinecone**, using **free-tier services only** (Gemini free tier for the LLM, a local free embedding model — no paid keys required except Pinecone's free plan).

## What this is

- `POST /ingest` — reads the corpus, chunks it, embeds it, upserts to Pinecone.
- `POST /ask` — takes a question, runs a LangGraph flow (retrieve → grade → answer with citations), returns JSON.
- Answers are grounded **only** in the retrieved chunks. If the docs don't cover the question, the API says so instead of guessing — even if the retrieval step returns some loosely related chunks.

See [`docs/langgraph.md`](docs/langgraph.md) for the graph diagram and node-by-node explanation.

---

## 1. Setup

### Requirements

- Python 3.10+
- A free [Pinecone](https://app.pinecone.io) account (free tier is enough)
- A free [Google AI Studio](https://aistudio.google.com/apikey) API key (Gemini free tier)

### Install

```bash
git clone https://github.com/ROKIGHOSH/legixo-qa.git
cd legixo-qa
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment variables

```bash
copy .env.example .env        # macOS/Linux: cp .env.example .env
```

Then edit `.env` and fill in the two real values you need:

| Variable | Where to get it |
|---|---|
| `PINECONE_API_KEY` | https://app.pinecone.io → API Keys |
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey (free, no credit card) |

Everything else in `.env.example` has a sensible default and does **not** need to be changed to run the project. See the full variable list and what each one does directly in `.env.example`.

**Never commit your real `.env` file** — only `.env.example` (with dummy values) is checked into git.

---

## 2. Ingest the corpus

The sample corpus lives in `gen_ai_takehome_sample_corpus/` (6 fictional legal-style `.md` files + a README). You can run ingest either as a script or via the API — both do the same thing.

**Option A — script:**
```bash
python -m app.ingest
```

**Option B — API route** (after starting the server, see step 3):
```bash
curl -X POST http://localhost:8000/ingest
```

Both print/return how many files and chunks were processed.

### What happens if you run ingest twice?

**Nothing bad.** Every ingest run first **deletes the entire Pinecone namespace** (`PINECONE_NAMESPACE`, default `legixo-default`), then re-chunks and re-upserts everything with deterministic chunk ids (`"<filename>::chunk<N>"`). So re-running ingest is idempotent — you always end up with exactly one copy of each chunk, never duplicates, and stale chunks from deleted/edited source files are cleared out automatically. This is implemented in `app/vector_store.py::clear_namespace` and called at the top of `app/ingest.py::run_ingest`.

If you want to keep multiple corpora side by side instead of overwriting, change `PINECONE_NAMESPACE` in `.env` before running ingest again — each namespace is independent within the same index.

---

## 3. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

Server docs (interactive Swagger UI) are then available at http://localhost:8000/docs

---

## 4. Call the ask endpoint

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the notice period for Priya Nambiar?\", \"include_trace\": true}"
```

Example response:

```json
{
  "answer": "Priya Nambiar's employment agreement specifies that either party may end the agreement by giving 60 days written notice.",
  "citations": [
    {
      "chunk_id": "02_employment_agreement_excerpt.md::chunk0",
      "source_file": "02_employment_agreement_excerpt.md",
      "score": 0.3113
    }
  ],
  "trace": [
    "retrieve: got 5 chunk(s) from Pinecone",
    "grade_chunks: scores=[0.3165, 0.3113, 0.3087, 0.21, 0.193] (threshold 0.2) -> 4/5 passed -> GOOD path",
    "answer_good: LLM named 1 source id(s), 1 matched real retrieved chunks"
  ]
}
```

Only one citation appears even though 4 chunks cleared the retrieval score threshold — that's intentional. See "Why citations are trustworthy" below.

### A question the docs cannot answer

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the maximum penalty for late filing of a criminal appeal?\"}"
```

```json
{
  "answer": "I cannot find this in the provided documents. The corpus does not contain information that answers this question.",
  "citations": []
}
```

(This corpus has nothing about criminal appeals — only civil/commercial matters, employment, leases, and a fictional statute excerpt. See `eval/self_test.json` question #12 for this exact check.)

---

## 5. Video walkthrough

**Video: https://drive.google.com/drive/folders/1rPsAZeZQN1cDyuvMJKJhpVpWgB9KvtCf?usp=sharing**

The video covers: install → `pip install -r requirements.txt` → ingest → start the API → call `/ask` with a few good questions (with citations shown) → one question the docs cannot answer → a walk through `app/graph.py` / `docs/langgraph.md` pointing at the LangGraph nodes.

---

## Project layout

```
legixo-qa/
├── app/
│   ├── main.py          # FastAPI app, /ask and /ingest routes
│   ├── graph.py          # LangGraph flow (retrieve -> grade -> answer)
│   ├── ingest.py          # ingest pipeline (chunk -> embed -> upsert)
│   ├── chunking.py        # paragraph-aware chunker with overlap
│   ├── embeddings.py      # local, free sentence-transformers embeddings
│   ├── llm.py              # Gemini (free tier) wrapper for answer generation
│   ├── vector_store.py     # Pinecone client wrapper (real service, not a fake)
│   └── config.py           # reads all settings from .env
├── docs/
│   └── langgraph.md        # graph diagram + node explanations
├── eval/
│   └── self_test.json      # ~12 self-test Q&A pairs incl. one out-of-corpus check
├── gen_ai_takehome_sample_corpus/   # the fictional legal-notes corpus
├── .env.example
├── requirements.txt
└── README.md
```

## Design notes / what you must use, satisfied

- **Python 3.10+** ✅
- **LangGraph (`StateGraph`)** — real graph with 4 nodes and a conditional branch, not one giant LLM call. See `app/graph.py`.
- **Pinecone**, real service via the official Python client — not an in-memory fake. See `app/vector_store.py`.
- **HTTP API only for Q&A** — `POST /ask` via FastAPI, no CLI for asking questions. Ingestion is available both as a documented script (`python -m app.ingest`) and an API route (`POST /ingest`), so reviewers can load the corpus without a custom REPL.
- **Branch in the graph**: `grade_chunks` routes to `answer_good` or `answer_not_found` based on retrieval score, and acts as a coarse first-pass filter.
- **Max step / loop guard**: every node increments a `steps` counter; `MAX_GRAPH_STEPS` (default 6) forces the not-found path if ever exceeded, so the graph cannot spin forever.
- **Free-tier only**: Gemini (`gemini-2.5-flash` by default) for generation, local `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings — no OpenAI key required. Only Pinecone needs a (free-tier) account.

### Why citations are trustworthy (no fake citations)

This is a small, 7-file corpus, so cosine similarity scores from the embedding model cluster close together (typically 0.19–0.45) — an absolute score threshold alone isn't a reliable "is this chunk actually relevant" signal on its own. To avoid citing chunks that merely scored above the threshold but aren't actually about the question:

1. `grade_chunks` does a coarse first pass using `MIN_GOOD_CHUNK_SCORE` (default `0.2`) just to decide whether to attempt an answer at all vs. give up immediately.
2. All chunks that pass are handed to the LLM, but the LLM is required to name exactly which `chunk_id`s it actually relied on (`SOURCES: ...`) in a structured response.
3. The API only returns citations for chunk ids the LLM explicitly named **and** that are real, retrieved chunks (never invented ids).
4. If the LLM judges that none of the retrieved chunks actually answer the question, it responds with a literal `NOT_FOUND` signal, which the app converts into the standard "I cannot find this in the provided documents" message with **zero citations** — even if the retrieval step returned some loosely related chunks. This is what correctly handles out-of-corpus questions (see `docs/langgraph.md`).

## What I skipped / didn't build

- Hybrid search / reranking (listed as optional "extras" in the brief) — not implemented, straight cosine-similarity vector search only.
- LangSmith tracing — not wired up; the graph exposes its own lightweight `trace` list in the response instead (`include_trace: true`).
- Authentication on the API — out of scope for a local take-home reviewer.
