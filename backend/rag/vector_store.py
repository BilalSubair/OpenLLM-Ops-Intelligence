import chromadb

from rag.embeddings import generate_embedding

client = chromadb.PersistentClient(
    path="../chroma_db"
)

collection = client.get_or_create_collection(
    name="infrastructure_knowledge"
)


def add_document(text: str, document_id: str):

    embedding = generate_embedding(text)

    collection.add(
        ids=[document_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "source": "initial_troubleshooting_guide",
                "category": "gpu"
            }
        ]
    )

    return {
        "message": "Document stored successfully",
        "id": document_id
    }


def search_documents(question: str, results_count: int = 3):

    query_embedding = generate_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=results_count
    )

    return results
