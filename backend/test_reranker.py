
from sentence_transformers import CrossEncoder

print("Loading BGE reranker model...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker model loaded successfully!")

question = "How can I reduce GPU memory consumption during LLM inference?"

documents = [
    "GPU memory stores model weights and activations.",
    "Reducing batch size and context length can reduce GPU memory consumption.",
    "Python is a programming language used for software development.",
    "The KV cache consumes GPU memory during LLM inference."
]

pairs = [
    [question, document]
    for document in documents
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(documents, scores),
    key=lambda item: float(item[1]),
    reverse=True
)

print("\nRERANKING RESULTS")

for rank, (document, score) in enumerate(
    ranked_results,
    start=1
):
    print(f"\nRank: {rank}")
    print(f"Score: {float(score):.4f}")
    print(f"Document: {document}")

