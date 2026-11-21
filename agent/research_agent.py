import json
from pathlib import Path
from typing import TypedDict
import re
import numpy as np
from langgraph.graph import END, START, StateGraph
from pdf_index import build_pdf_index

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
    snippets: list[dict]
    pdf_paths : list[str]
    overview_only: bool
    paper_index: tuple[list[dict], np.ndarray]


def rank_evidence_snippets(question, evidence, reranker):
    candidates = []
    seen = set()

    for passage_id, passage in enumerate(evidence, start=1):
        for part in re.split(r"(?<=[.!?])\s+", passage["text"]):
            text = part.strip()
            key = " ".join(text.split())

            if not key or key in seen:
                continue

            seen.add(key)
            candidates.append({"text": text, "passage_id": passage_id})

    if not candidates:
        return []

    return rerank_chunks(reranker, question, candidates)[:8]



def research_agent(state: ResearchState) -> dict:
    model = load_embedding_model()
    pdf_paths = state.get("pdf_paths", [])
    paper_index = state.get("paper_index")

    if paper_index is not None:
        chunks, vectors = paper_index
    elif pdf_paths:
        chunks = []
        vector_batches = []

        for pdf_path in pdf_paths:
            paper_chunks, paper_vectors = build_pdf_index(Path(pdf_path), model)
            chunks.extend(paper_chunks)
            vector_batches.append(paper_vectors)

        vectors = np.concatenate(vector_batches, axis=0)
        print(f"Indexed {len(pdf_paths)} PDF(s): {len(chunks)} chunks.")
    else:
        chunks = json.loads((PROJECT / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(PROJECT / "chunk_embeddings.npy", allow_pickle=False)

    if state.get("overview_only"):
        keep = [i for i, chunk in enumerate(chunks) if chunk["page"] <= 2]
        if not keep:
            return {"evidence": [], "snippets": []}

        chunks = [chunks[i] for i in keep]
        vectors = vectors[keep]

    question = state["question"]
    candidates = retrieve_top_chunks(model, question, chunks, vectors, top_k=20)
    reranker = load_reranker()
    evidence = rerank_chunks(reranker, question, candidates)[:6]
    snippets = rank_evidence_snippets(question, evidence, reranker)
    return {"evidence": evidence, "snippets": snippets}

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
