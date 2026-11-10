from rag.ingestion import ingest_document
from rag.vector_store import collection

document = """
CUDA is a parallel computing platform developed by NVIDIA.

GPU memory, also called VRAM, stores model weights,
activations, and other data required for inference.

CUDA Out of Memory happens when the GPU does not
have enough available memory for an operation.

Common solutions include reducing batch size,
reducing context length, using model quantization,
and selecting a smaller model.

For LLM inference, GPU memory is also required
for the KV cache, model weights, and activations.

Reducing the context length can reduce KV cache
memory consumption.
"""

result = ingest_document(
    text=document,
    document_id="gpu_guide_001",
    source="gpu_troubleshooting.txt",
    category="gpu"
)

print("\nIngestion result:")
print(result)

print("\nTotal documents in ChromaDB:")
print(collection.count())

print("\nStored chunk IDs:")
print(collection.get()["ids"])
