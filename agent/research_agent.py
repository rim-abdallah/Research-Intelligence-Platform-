import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from rag_utils import (
    load_embedding_model,
    load_reranker,
    rerank_chunks,
    retrieve_top_chunks,
)

PROJECT = Path(__file__).resolve().parents[1]


class ResearchState(TypedDict, total=False):
    question: str
    evidence: list[dict]


def research_agent(state: ResearchState) -> dict:
    chunks = json.loads((PROJECT / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(PROJECT / "chunk_embeddings.npy", allow_pickle=False)

    question = state["question"]
    candidates = retrieve_top_chunks(
        load_embedding_model(), question, chunks, vectors, top_k=20
    )
    ranked = rerank_chunks(load_reranker(), question, candidates)
    return {"evidence": ranked[:6]}


if __name__ == "__main__":
    graph = StateGraph(ResearchState)
    graph.add_node("research", research_agent)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    workflow = graph.compile()

    question = input("Ask about LightGCN: ").strip()
    if not question:
        raise SystemExit("Enter a question first.")

    result = workflow.invoke({"question": question})
    print(f"Returned {len(result['evidence'])} passages.")
    for passage in result["evidence"]:
        print(f"{passage['source']} | PDF page {passage['page']}")
    if result["evidence"]:
        print("\nTop passage:", result["evidence"][0]["text"])
