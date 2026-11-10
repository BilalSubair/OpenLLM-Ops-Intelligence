
from rag.hybrid_search import hybrid_search
from rag.reranker import rerank_documents


question = "How can I fix CUDA Out of Memory errors during LLM inference?"

print("\n1. Running Hybrid Search...")

hybrid_results = hybrid_search(
    question,
    results_count=5
)

print(f"Retrieved {len(hybrid_results)} documents")


print("\n2. Running BGE Reranker...")

reranked_results = rerank_documents(
    question=question,
    documents=hybrid_results,
    top_k=3
)


print("\n3. FINAL RERANKED RESULTS")

for result in reranked_results:
    print("\nDocument ID:", result["id"])
    print("Rerank Score:", result["rerank_score"])
    print("Document:", result["document"])

