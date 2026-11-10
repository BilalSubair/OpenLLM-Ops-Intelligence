"""
BGE Cross-Encoder Reranker Module.
Re-ranks candidate document chunks against user questions for high precision.
Uses lazy loading and graceful offline fallback.
"""

from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

RERANKER_MODEL = "BAAI/bge-reranker-base"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None and CrossEncoder:
        try:
            _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        except Exception:
            _reranker = False
    return _reranker if _reranker is not False else None


# --------------------------------------------------
# DOCUMENT RERANKING
# --------------------------------------------------

def rerank_documents(
    question: str,
    documents: list,
    top_k: int = 3
):
    if not documents:
        return []

    model = get_reranker()

    if model:
        # Create question-document pairs
        pairs = [
            [question, document["document"]]
            for document in documents
        ]

        scores = model.predict(pairs)

        ranked_results = []
        for document, score in zip(documents, scores):
            ranked_results.append({
                "id": document["id"],
                "document": document["document"],
                "rerank_score": float(score)
            })

        ranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )
        return ranked_results[:top_k]

    # Offline / Graceful Fallback: retain existing scores (RRF or dense)
    results = []
    for rank, doc in enumerate(documents[:top_k], 1):
        score = doc.get("rrf_score") or doc.get("score") or round(1.0 / rank, 3)
        results.append({
            "id": doc["id"],
            "document": doc["document"],
            "rerank_score": float(score)
        })
    return results
