
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# RERANKER CONFIGURATION
# --------------------------------------------------

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Load the model once when this module is imported
print("Loading BGE reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    max_length=512
)

print("BGE reranker loaded successfully!")


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

    # Create question-document pairs
    pairs = [
        [question, document["document"]]
        for document in documents
    ]

    # Calculate relevance scores
    scores = reranker.predict(pairs)

    # Attach scores to documents
    ranked_results = []

    for document, score in zip(documents, scores):
        ranked_results.append({
            "id": document["id"],
            "document": document["document"],
            "rerank_score": float(score)
        })

    # Sort by relevance score
    ranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # Return top relevant documents
    return ranked_results[:top_k]

