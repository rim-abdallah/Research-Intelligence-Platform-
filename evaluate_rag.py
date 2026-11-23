import csv
import json
from pathlib import Path
from argparse import ArgumentParser
from pdf_index import build_pdf_index

import numpy as np

from rag_utils import (
    load_embedding_model,
    load_reranker,
    rerank_chunks,
    retrieve_top_chunks,
)


K_VALUES = (1, 3, 5, 10, 20)


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

    args = parser.parse_args()

    with args.questions.open(encoding="utf-8") as f:
        questions = list(csv.DictReader(f))


    print("\nLoading embedding model...")
    model = load_embedding_model()

    if args.pdf is not None:
        print("Evaluation PDF:", args.pdf.name)
        chunks, chunk_embeddings = build_pdf_index(args.pdf, model)
    else:
        with Path("chunks.json").open(encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_embeddings = np.load("chunk_embeddings.npy", allow_pickle=False)


    print("Loading reranker...")
    reranker = load_reranker()

    print("Number of evaluation questions:", len(questions))
    print("Loaded chunks:", len(chunks))
    print("Embeddings shape:", chunk_embeddings.shape)

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

        retrieved = retrieve_top_chunks(
            model, question, chunks, chunk_embeddings, top_k=20
        )
        print("Original retrieved chunks:", [item["index"] for item in retrieved])

        reranked = rerank_chunks(reranker, question, retrieved)
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
