from rag.pipeline import process_document
from rag.vector_store import collection

result = process_document(
    "data/documents/sample.txt"
)

print("\nPipeline result:")
print(result)

print("\nTotal records in ChromaDB:")
print(collection.count())

print("\nStored IDs:")
print(collection.get()["ids"])

