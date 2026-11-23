import csv
import json
import re
from pathlib import Path
from argparse import ArgumentParser
from pdf_index import build_pdf_index, extract_pdf_chunks

import numpy as np

from rag_utils import (
    load_embedding_model,
    load_reranker,
    rerank_chunks,
    retrieve_top_chunks,
)


K_VALUES = (1, 3, 5, 10, 20)
CANDIDATE_LIMIT = max(K_VALUES)
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text):
    return WORD.findall(text.casefold())


def build_bm25_retriever(chunks):
    from rank_bm25 import BM25Okapi

    index = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def retrieve(question, top_k=CANDIDATE_LIMIT):
        scores = index.get_scores(tokenize(question))
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        matches = (i for i in order[:top_k] if scores[i] > 0)
        return [
            {**chunks[i], "index": i, "rank": rank, "score": float(scores[i])}
            for rank, i in enumerate(matches, start=1)
        ]

    return retrieve


def find_gold_chunks(gold_evidence, chunks):
    phrases = [
        phrase.strip().lower()
        for phrase in gold_evidence.split(";")
        if phrase.strip()
    ]
    return {
        index
        for index, chunk in enumerate(chunks)
        if any(phrase in chunk["text"].lower() for phrase in phrases)
    }


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("lightgcn_eval_questions.csv"),
    )
    parser.add_argument("--pdf", type=Path)
    parser.add_argument(
        "--retrieval",
        choices=("dense", "bm25"),
        default="dense",
        help="Candidate search method; both modes use the same reranker.",
    )

    args = parser.parse_args()

    with args.questions.open(encoding="utf-8") as f:
        questions = list(csv.DictReader(f))


    print("Candidate retrieval:", args.retrieval)
    if args.pdf is not None:
        print("Evaluation PDF:", args.pdf.name)

    if args.retrieval == "bm25":
        if args.pdf is not None:
            chunks = extract_pdf_chunks(args.pdf)
        else:
            with Path("chunks.json").open(encoding="utf-8") as f:
                chunks = json.load(f)
        retrieve = build_bm25_retriever(chunks)
    else:
        print("Loading embedding model...")
        model = load_embedding_model()
        if args.pdf is not None:
            chunks, chunk_embeddings = build_pdf_index(args.pdf, model)
        else:
            with Path("chunks.json").open(encoding="utf-8") as f:
                chunks = json.load(f)
            chunk_embeddings = np.load("chunk_embeddings.npy", allow_pickle=False)
        print("Embeddings shape:", chunk_embeddings.shape)

        def retrieve(question):
            return retrieve_top_chunks(
                model, question, chunks, chunk_embeddings, top_k=CANDIDATE_LIMIT
            )

    print("Loading reranker...")
    reranker = load_reranker()

    print("Number of evaluation questions:", len(questions))
    print("Loaded chunks:", len(chunks))

    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_rank_sum = 0.0
    evaluated_questions = 0

    for i, row in enumerate(questions, start=1):
        question = row["question"]
        print(f"\n{'=' * 70}")
        print(f"Question {i}: {question}")
        print("Expected answer:", row["expected_answer"])

        if row["answer_present"].strip().lower() == "no":
            print("Negative question: no gold evidence.")
            continue

        gold_chunks = find_gold_chunks(row["gold_evidence"], chunks)
        print("Gold evidence:", row["gold_evidence"])
        print("Current gold chunks:", sorted(gold_chunks))

        if not gold_chunks:
            print("WARNING: No chunk contains the gold evidence.")
            continue

        retrieved = retrieve(question)
        print("Original retrieved chunks:", [item["index"] for item in retrieved])

        reranked = rerank_chunks(reranker, question, retrieved) if retrieved else []
        indices = [item["index"] for item in reranked]

        print("Reranked chunks:", indices)
        print("\nTop reranked results:")
        for item in reranked[:5]:
            print(
                f"Rank {item['rerank_rank']} | Chunk {item['index']} | "
                f"Rerank score: {item['rerank_score']:.3f}"
            )

        evaluated_questions += 1

        for k in K_VALUES:
            if any(index in gold_chunks for index in indices[:k]):
                hits_at_k[k] += 1

        first_relevant_rank = next(
            (
                rank
                for rank, index in enumerate(indices, start=1)
                if index in gold_chunks
            ),
            None,
        )

        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

        print("First relevant rank after reranking:", first_relevant_rank)

    print("\n" + "=" * 70)
    print("Retrieval evaluation after reranking")
    print("=" * 70)
    print("Evaluated questions:", evaluated_questions)

    if evaluated_questions:
        for k in K_VALUES:
            print(f"Hit@{k}: {hits_at_k[k] / evaluated_questions:.3f}")
        print(f"MRR: {reciprocal_rank_sum / evaluated_questions:.3f}")


if __name__ == "__main__":
    main()
