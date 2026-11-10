from rag.bm25_search import bm25_search

question = "CUDA GPU memory out of memory"

results = bm25_search(question, results_count=3)

print("\nBM25 Search Results:")

for result in results:
    print("\nDocument ID:", result["id"])
    print("BM25 Score:", result["score"])
    print("Document:", result["document"])
