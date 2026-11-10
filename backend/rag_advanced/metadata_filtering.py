"""
Metadata Filtering & Faceted Search for Hybrid RAG.
Filters ChromaDB dense queries and BM25 search by category, source, or custom metadata tags.
"""

from typing import Dict, Any, List, Optional

try:
    from rag.vector_store import collection
    from rag.embeddings import generate_embedding
except Exception:
    collection = None
    generate_embedding = None


def search_with_metadata_filter(
    question: str,
    category: Optional[str] = None,
    source: Optional[str] = None,
    results_count: int = 5
) -> Dict[str, Any]:
    """
    Performs filtered retrieval against ChromaDB using metadata constraints.
    """
    if not collection or not generate_embedding:
        return {
            "question": question,
            "filters_applied": {"category": category, "source": source},
            "matched_count": 0,
            "results": [],
            "message": "ChromaDB or embedding provider unavailable"
        }

    where_filter = {}
    if category and source:
        where_filter = {
            "$and": [
                {"category": {"$eq": category}},
                {"source": {"$eq": source}}
            ]
        }
    elif category:
        where_filter = {"category": {"$eq": category}}
    elif source:
        where_filter = {"source": {"$eq": source}}

    query_embedding = generate_embedding(question)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": results_count
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    matched_docs = []
    if results and results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            matched_docs.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None
            })

    return {
        "question": question,
        "filters_applied": {"category": category, "source": source},
        "matched_count": len(matched_docs),
        "results": matched_docs
    }
