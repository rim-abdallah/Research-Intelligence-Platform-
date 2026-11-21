import csv
import json
from pathlib import Path

import numpy as np

from rag_utils import (
    load_embedding_model,
    load_reranker,
    retrieve_top_chunks,
    rerank_chunks
)


# File paths
eval_path = Path("lightgcn_eval_questions.csv")
chunks_path = Path("chunks.json")
embeddings_path = Path("chunk_embeddings.npy")


# Load evaluation questions
with open(eval_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    questions = list(reader)


# Load saved chunks
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)


# Load saved embeddings
chunk_embeddings = np.load(embeddings_path)


# Load models
print("\nLoading embedding model...")
model = load_embedding_model()

print("Loading reranker...")
reranker = load_reranker()


print("Number of evaluation questions:", len(questions))
print("Loaded chunks:", len(chunks))
print("Embeddings shape:", chunk_embeddings.shape)


# Metrics
recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0
recall_at_10 = 0
recall_at_20 = 0

reciprocal_rank_sum = 0
evaluated_questions = 0


def find_gold_chunks(gold_evidence, chunks):
    """
    Find all current chunks containing one of the gold evidence phrases.
    """
    phrases = [
        phrase.strip().lower()
        for phrase in gold_evidence.split(";")
        if phrase.strip()
    ]

    gold_indices = set()

    for index, chunk in enumerate(chunks):
        chunk_text = chunk["text"].lower()

        for phrase in phrases:
            if phrase in chunk_text:
                gold_indices.add(index)
                break

    return gold_indices


# Evaluate retrieval
for i, row in enumerate(questions, start=1):

    question = row["question"]
    answer_present = row["answer_present"].strip().lower()

    print(f"\n{'=' * 70}")
    print(f"Question {i}: {question}")
    print("Expected answer:", row["expected_answer"])

    # Skip negative questions
    if answer_present == "no":
        print("Negative question: no gold evidence.")
        continue

    gold_evidence = row["gold_evidence"]

    gold_chunks = find_gold_chunks(
        gold_evidence,
        chunks
    )

    print("Gold evidence:", gold_evidence)
    print("Current gold chunks:", sorted(gold_chunks))

    if not gold_chunks:
        print("WARNING: No chunk contains the gold evidence.")
        continue

    # First-stage retrieval
    retrieved = retrieve_top_chunks(
        model,
        question,
        chunks,
        chunk_embeddings,
        top_k=20
    )

    original_indices = [
        result["index"]
        for result in retrieved
    ]

    print("Original retrieved chunks:", original_indices)

    # Rerank the same 20 candidates
    reranked = rerank_chunks(
        reranker,
        question,
        retrieved
    )

    retrieved_indices = [
        result["index"]
        for result in reranked
    ]

    print("Reranked chunks:", retrieved_indices)

    # Optional: show reranker scores
    print("\nTop reranked results:")

    for result in reranked[:5]:
        print(
            f"Rank {result['rerank_rank']} | "
            f"Chunk {result['index']} | "
            f"Rerank score: {result['rerank_score']:.3f}"
        )

    evaluated_questions += 1

    # Recall@1
    if any(
        index in gold_chunks
        for index in retrieved_indices[:1]
    ):
        recall_at_1 += 1

    # Recall@3
    if any(
        index in gold_chunks
        for index in retrieved_indices[:3]
    ):
        recall_at_3 += 1

    # Recall@5
    if any(
        index in gold_chunks
        for index in retrieved_indices[:5]
    ):
        recall_at_5 += 1

    # Recall@10
    if any(
        index in gold_chunks
        for index in retrieved_indices[:10]
    ):
        recall_at_10 += 1

    # Recall@20
    if any(
        index in gold_chunks
        for index in retrieved_indices[:20]
    ):
        recall_at_20 += 1

    # Reciprocal Rank
    first_relevant_rank = None

    for rank, index in enumerate(
        retrieved_indices,
        start=1
    ):
        if index in gold_chunks:
            first_relevant_rank = rank
            break

    if first_relevant_rank is not None:
        reciprocal_rank_sum += 1 / first_relevant_rank

    print("First relevant rank after reranking:", first_relevant_rank)


# Final metrics
print("\n" + "=" * 70)
print("Retrieval evaluation after reranking")
print("=" * 70)

print("Evaluated questions:", evaluated_questions)

if evaluated_questions > 0:

    print(
        f"Recall@1: "
        f"{recall_at_1 / evaluated_questions:.3f}"
    )

    print(
        f"Recall@3: "
        f"{recall_at_3 / evaluated_questions:.3f}"
    )

    print(
        f"Recall@5: "
        f"{recall_at_5 / evaluated_questions:.3f}"
    )

    print(
        f"Recall@10: "
        f"{recall_at_10 / evaluated_questions:.3f}"
    )

    print(
        f"Recall@20: "
        f"{recall_at_20 / evaluated_questions:.3f}"
    )

    mrr = reciprocal_rank_sum / evaluated_questions

    print(f"MRR: {mrr:.3f}")