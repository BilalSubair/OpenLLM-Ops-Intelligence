from rag.vector_store import search_documents
from rag.bm25_search import bm25_search


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
    results_count=3
):
    fused_scores = {}
    documents = {}

    # Process Dense Search results
    for rank, document_id in enumerate(
        dense_results["ids"][0],
        start=1
    ):
        score = 1 / (k + rank)

        fused_scores[document_id] = (
            fused_scores.get(document_id, 0) + score
        )

        documents[document_id] = dense_results["documents"][0][rank - 1]

    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        document_id = result["id"]
        score = 1 / (k + rank)

        fused_scores[document_id] = (
            fused_scores.get(document_id, 0) + score
        )

        documents[document_id] = result["document"]

    # Sort by combined RRF score
    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {
            "id": document_id,
            "document": documents[document_id],
            "rrf_score": score
        }
        for document_id, score in ranked_documents[:results_count]
    ]


def hybrid_search(question: str, results_count: int = 3):

    dense_results = search_documents(
        question,
        results_count=results_count
    )

    bm25_results = bm25_search(
        question,
        results_count=results_count
    )

    return reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        results_count=results_count
    )
