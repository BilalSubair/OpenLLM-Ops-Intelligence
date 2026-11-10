from rag.chunking import split_text

sample_text = """
CUDA is a parallel computing platform developed by NVIDIA.

GPU memory, also called VRAM, stores model weights,
activations, and other data required for inference.

CUDA Out of Memory happens when the GPU does not
have enough available memory for an operation.

Solutions include reducing batch size, reducing
context length, and using model quantization.
"""

chunks = split_text(
    sample_text,
    chunk_size=150,
    chunk_overlap=30
)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
