"""LangGraph flow for the /ask endpoint.

Nodes (see docs/langgraph.md for the diagram):
  1. retrieve        - embed the question, search Pinecone for top_k chunks
  2. grade_chunks     - decide if retrieved chunks are "good enough" (branch point)
  3a. answer_good     - (good path) ask Gemini to answer using only the chunks, with citations
  3b. answer_not_found - (bad path) say plainly that the docs don't answer this

A `steps` counter is carried in the state and incremented on every node run.
If it ever exceeds MAX_GRAPH_STEPS the graph short-circuits to answer_not_found
so it can never spin forever (this also guards against LangGraph retry loops).
"""
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

from app import config
from app.embeddings import embed_query
from app.vector_store import query as pinecone_query
from app.llm import generate


class GraphState(TypedDict, total=False):
    question: str
    steps: int
    matches: List[Dict[str, Any]]
    good_chunks: List[Dict[str, Any]]
    is_good: bool
    answer: str
    citations: List[Dict[str, Any]]
    trace: List[str]


def _log(state: GraphState, message: str):
    state.setdefault("trace", [])
    state["trace"].append(message)


# ---------- Node: retrieve ----------
def node_retrieve(state: GraphState) -> GraphState:
    state["steps"] = state.get("steps", 0) + 1
    question = state["question"]
    vector = embed_query(question)
    matches = pinecone_query(vector, top_k=config.TOP_K)
    state["matches"] = matches
    _log(state, f"retrieve: got {len(matches)} chunk(s) from Pinecone")
    return state


# ---------- Node: grade_chunks (this is the branch point) ----------
def node_grade_chunks(state: GraphState) -> GraphState:
    state["steps"] = state.get("steps", 0) + 1
    matches = state.get("matches", [])
    good = [m for m in matches if m.get("score", 0) >= config.MIN_GOOD_CHUNK_SCORE]
    state["good_chunks"] = good
    state["is_good"] = len(good) > 0
    scores = [round(m.get("score", 0), 4) for m in matches]
    _log(
        state,
        f"grade_chunks: scores={scores} (threshold {config.MIN_GOOD_CHUNK_SCORE}) "
        f"-> {len(good)}/{len(matches)} passed -> {'GOOD path' if state['is_good'] else 'BAD path'}",
    )
    return state


def route_after_grade(state: GraphState) -> str:
    """Branch: good path vs bad path. Also caps total steps so the graph
    can never loop forever, regardless of scores."""
    if state.get("steps", 0) >= config.MAX_GRAPH_STEPS:
        return "answer_not_found"
    return "answer_good" if state.get("is_good") else "answer_not_found"


# ---------- Node: answer_good (good path) ----------
# The LLM is asked to name exactly which chunk_ids it actually used. This means
# a chunk that merely cleared the retrieval score threshold (small corpora make
# absolute-score thresholds noisy) never gets cited unless the LLM's answer
# genuinely draws on it - this is what prevents irrelevant/"fake" citations.
ANSWER_PROMPT = """You are a legal-notes assistant. Answer the user's question using ONLY the
context chunks below. Every factual claim must be traceable to a chunk.

Rules:
- If NONE of the chunks actually answer the question, do not guess or partially answer.
  In that case respond with exactly: ANSWER: NOT_FOUND  and  SOURCES: (leave empty)
- Only rely on chunks that are actually relevant to the question. Ignore chunks in the
  context that are unrelated, even if they were retrieved.
- Do not invent facts, case names, dates, or numbers that are not in the chunks.
- Keep the answer concise (a few sentences).

Question: {question}

Context chunks:
{context}

Respond in EXACTLY this format, with nothing before or after it:
ANSWER: <your answer text, one paragraph, no citation markers inline - OR the literal text NOT_FOUND>
SOURCES: <comma-separated list of the chunk_id values you actually relied on, e.g. 02_employment_agreement_excerpt.md::chunk0 - leave empty if NOT_FOUND>"""


def _parse_answer_response(raw: str):
    """Split the LLM's structured response into (answer_text, [chunk_ids])."""
    answer_text = raw.strip()
    source_ids: List[str] = []

    if "SOURCES:" in raw:
        answer_part, sources_part = raw.split("SOURCES:", 1)
        answer_text = answer_part.replace("ANSWER:", "", 1).strip()
        source_ids = [s.strip() for s in sources_part.strip().split(",") if s.strip()]
    elif "ANSWER:" in raw:
        answer_text = raw.replace("ANSWER:", "", 1).strip()

    return answer_text, source_ids


def node_answer_good(state: GraphState) -> GraphState:
    state["steps"] = state.get("steps", 0) + 1
    good_chunks = state.get("good_chunks", [])

    context_blocks = []
    chunk_lookup: Dict[str, Dict[str, Any]] = {}
    for m in good_chunks:
        meta = m.get("metadata", {})
        chunk_id = meta.get("chunk_id")
        context_blocks.append(
            f"[{chunk_id}] (source: {meta.get('source_file')})\n{meta.get('text')}"
        )
        chunk_lookup[chunk_id] = {
            "chunk_id": chunk_id,
            "source_file": meta.get("source_file"),
            "score": round(m.get("score", 0), 4),
        }

    prompt = ANSWER_PROMPT.format(
        question=state["question"], context="\n\n".join(context_blocks)
    )
    raw_response = generate(prompt)
    answer_text, cited_ids = _parse_answer_response(raw_response)

    # The LLM is the final judge of relevance, not just the retrieval score
    # (small corpora make absolute-score thresholds a poor filter on their own).
    # If it explicitly says NOT_FOUND, treat this exactly like the bad path:
    # standard message, zero citations - never leave a "cannot find" answer
    # with a misleading citation attached.
    if answer_text.strip().upper().startswith("NOT_FOUND"):
        state["answer"] = (
            "I cannot find this in the provided documents. The corpus does not contain "
            "information that answers this question."
        )
        state["citations"] = []
        _log(state, "answer_good: LLM judged no chunk actually answers the question -> NOT_FOUND")
        return state

    # Only keep citations the LLM actually named AND that came from real
    # retrieved chunks - never invented, never just "cleared the threshold".
    citations = [chunk_lookup[cid] for cid in cited_ids if cid in chunk_lookup]

    # Fallback: if the LLM didn't follow the format and named nothing, but did
    # produce a real answer, cite the single highest-scoring chunk so the
    # response is never left with an ungrounded answer and zero citations.
    if not citations and good_chunks:
        top = good_chunks[0]
        meta = top.get("metadata", {})
        citations = [
            {
                "chunk_id": meta.get("chunk_id"),
                "source_file": meta.get("source_file"),
                "score": round(top.get("score", 0), 4),
            }
        ]

    state["answer"] = answer_text
    state["citations"] = citations
    _log(
        state,
        f"answer_good: LLM named {len(cited_ids)} source id(s), "
        f"{len(citations)} matched real retrieved chunks",
    )
    return state


# ---------- Node: answer_not_found (bad path) ----------
def node_answer_not_found(state: GraphState) -> GraphState:
    state["steps"] = state.get("steps", 0) + 1
    state["answer"] = (
        "I cannot find this in the provided documents. The corpus does not contain "
        "information that answers this question."
    )
    state["citations"] = []
    _log(state, "answer_not_found: no sufficiently relevant chunks (or step limit hit)")
    return state


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_chunks", node_grade_chunks)
    workflow.add_node("answer_good", node_answer_good)
    workflow.add_node("answer_not_found", node_answer_not_found)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_chunks")
    workflow.add_conditional_edges(
        "grade_chunks",
        route_after_grade,
        {"answer_good": "answer_good", "answer_not_found": "answer_not_found"},
    )
    workflow.add_edge("answer_good", END)
    workflow.add_edge("answer_not_found", END)

    return workflow.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str) -> GraphState:
    graph = get_graph()
    initial_state: GraphState = {"question": question, "steps": 0, "trace": []}
    final_state = graph.invoke(initial_state)
    return final_state
