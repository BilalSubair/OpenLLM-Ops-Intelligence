from rank_bm25 import BM25Okapi
from rag.vector_store import collection


def tokenize(text: str):
    return text.lower().split()


def bm25_search(question: str, results_count: int = 3):

    # Get all stored documents from ChromaDB
    stored_data = collection.get()

    documents = stored_data["documents"]
    ids = stored_data["ids"]

    if not documents:
        return []

    # Tokenize stored documents
    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize user question
    tokenized_question = tokenize(question)

    # Calculate relevance scores
    scores = bm25.get_scores(tokenized_question)

    # Rank documents by relevance
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:results_count]:
        results.append({
            "id": ids[index],
            "document": documents[index],
            "score": float(scores[index])
        })

    return results
