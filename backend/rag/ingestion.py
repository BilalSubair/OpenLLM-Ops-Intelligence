from rag.chunking import split_text
from rag.embeddings import generate_embedding
from rag.vector_store import collection


def ingest_document(
    text: str,
    document_id: str,
    source: str,
    category: str
):
    # 1. Split document into chunks
    chunks = split_text(
        text,
        chunk_size=500,
        chunk_overlap=100
    )

    if not chunks:
        raise ValueError("Document contains no usable text")

    # 2. Generate embeddings for each chunk
    embeddings = [
        generate_embedding(chunk)
        for chunk in chunks
    ]

    # 3. Create unique IDs for each chunk
    chunk_ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    # 4. Store chunks in ChromaDB
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": source,
                "category": category,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )

    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "message": "Document ingested successfully"
    }
