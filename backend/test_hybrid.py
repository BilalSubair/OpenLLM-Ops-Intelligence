from rag.hybrid_search import hybrid_search

question = "CUDA GPU memory out of memory"

results = hybrid_search(
    question,
    results_count=3
)

print("\nHYBRID SEARCH RESULTS")

for result in results:
    print("\nDocument ID:", result["id"])
    print("RRF Score:", result["rrf_score"])
    print("Document:", result["document"])
