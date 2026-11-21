from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def load_reranker():
    return CrossEncoder(RERANKER_NAME, device="cpu")


def create_embeddings(model, texts, show_progress=False):
    return model.encode(texts, show_progress_bar=show_progress)


def retrieve_top_chunks(
    model,
    question,
    chunks,
    chunk_embeddings,
    top_k=6,
):
    question_embedding = create_embeddings(model, [question])
    scores = model.similarity(question_embedding, chunk_embeddings)[0]
    top_results = scores.topk(k=min(top_k, len(chunks)))

    retrieved = []

    for rank, index in enumerate(top_results.indices.tolist(), start=1):
        chunk = chunks[index]

        retrieved.append({
            "rank": rank,
            "index": index,
            "score": scores[index].item(),
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
        })

    return retrieved


def rerank_chunks(reranker, question, retrieved):
    pairs = [[question, result["text"]] for result in retrieved]
    rerank_scores = reranker.predict(pairs)

    for result, score in zip(retrieved, rerank_scores):
        result["rerank_score"] = float(score)

    reranked = sorted(
        retrieved,
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    for rank, result in enumerate(reranked, start=1):
        result["rerank_rank"] = rank

    return reranked
