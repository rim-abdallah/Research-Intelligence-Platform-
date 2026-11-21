from sentence_transformers import SentenceTransformer, CrossEncoder


# Embedding model
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reranker model
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_embedding_model():
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    return model


def load_reranker():
    reranker = CrossEncoder(
        RERANKER_NAME,
        device="cpu"
    )

    return reranker


def create_embeddings(model, texts, show_progress=False):
    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress
    )

    return embeddings


def retrieve_top_chunks(
    model,
    question,
    chunks,
    chunk_embeddings,
    top_k=6
):
    question_embedding = create_embeddings(
        model,
        [question]
    )

    scores = model.similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    top_results = scores.topk(
        k=min(top_k, len(chunks))
    )

    retrieved = []

    for rank, index in enumerate(
        top_results.indices.tolist(),
        start=1
    ):
        chunk = chunks[index]

        retrieved.append({
            "rank": rank,
            "index": index,
            "score": scores[index].item(),
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return retrieved


def rerank_chunks(
    reranker,
    question,
    retrieved
):
    # Create question-passage pairs
    pairs = [
        [question, result["text"]]
        for result in retrieved
    ]

    # Get reranker scores
    rerank_scores = reranker.predict(pairs)

    # Add reranker score to each result
    for result, score in zip(
        retrieved,
        rerank_scores
    ):
        result["rerank_score"] = float(score)

    # Sort from highest reranker score to lowest
    reranked = sorted(
        retrieved,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Add new rank after reranking
    for rank, result in enumerate(
        reranked,
        start=1
    ):
        result["rerank_rank"] = rank

    return reranked