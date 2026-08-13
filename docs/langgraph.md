# LangGraph map

The `/ask` endpoint runs a small `StateGraph` (defined in `app/graph.py`) with 4 nodes.

```
        ┌───────────┐
        │  retrieve │   embed the question, search Pinecone (top_k chunks)
        └─────┬─────┘
              │
        ┌─────▼──────┐
        │ grade_chunks│  branch point: are any chunks "good enough"?
        └─────┬──────┘   (score >= MIN_GOOD_CHUNK_SCORE, and steps < MAX_GRAPH_STEPS)
              │
     ┌────────┴─────────┐
     │                   │
  GOOD path          BAD path
     │                   │
┌────▼─────┐      ┌──────▼────────────┐
│answer_good│      │answer_not_found   │
│(Gemini    │      │("cannot find      │
│ writes    │      │  this in docs")   │
│ answer +  │      │                   │
│ citations)│      │                   │
└────┬─────┘      └──────┬────────────┘
     │                    │
     └─────────┬──────────┘
               ▼
              END
```

## Nodes

| Node | What it does |
|---|---|
| `retrieve` | Embeds the user's question (local `sentence-transformers` model) and queries Pinecone for the top `TOP_K` chunks (default 5). |
| `grade_chunks` | **Branch point.** Filters retrieved chunks to those scoring `>= MIN_GOOD_CHUNK_SCORE` (default 0.35 cosine similarity). If at least one chunk passes, routes to the *good* path; otherwise routes to the *bad* path. |
| `answer_good` | Builds a prompt containing only the good chunks (with `chunk_id` + `source_file` labels) and asks Gemini to answer using only that context. Returns the answer plus a `citations` list built directly from chunk metadata (not invented by the LLM). |
| `answer_not_found` | Returns a fixed "I cannot find this in the provided documents" message with an empty citations list. |

## Loop / step guard

Every node increments a `steps` counter in the graph state. The conditional edge
after `grade_chunks` checks `steps >= MAX_GRAPH_STEPS` (default 6) and, if hit,
forces the `answer_not_found` path regardless of chunk scores. Since the graph
has no cycles (each path is a straight line to `END`), this is a belt-and-suspenders
guard rather than something that fires in normal operation - but it means the
graph can never spin forever even if nodes were extended later to loop/retry.

## Why citations aren't left to the LLM

`citations` in the API response come from Pinecone match metadata
(`chunk_id`, `source_file`, `score`), not from parsing the LLM's text. This
avoids fake/hallucinated citations - the LLM only writes the answer prose,
and only ever sees the chunks that are actually cited.
