"""
RAG Retrieval Accuracy & Quality Evaluation Engine.
Computes Hit Rate@K, Mean Reciprocal Rank (MRR), Context Precision,
and Query-Context Relevancy.
"""

from typing import Dict, Any, List, Optional
import math


class RetrievalEvaluator:
    """
    Evaluates accuracy and performance of Dense, BM25, and Hybrid search pipelines.
    """

    @staticmethod
    def calculate_hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """
        Hit Rate = 1.0 if any ground truth document is in retrieved documents, else 0.0.
        """
        if not ground_truth_ids:
            return 1.0
        return 1.0 if any(gt in retrieved_ids for gt in ground_truth_ids) else 0.0

    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank (MRR): 1 / rank of the first relevant document.
        """
        if not ground_truth_ids:
            return 1.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        """
        Precision@K = (# relevant retrieved in top K) / K.
        """
        if k <= 0 or not retrieved_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
        return round(relevant_count / float(k), 4)

    @staticmethod
    def calculate_context_relevance(query: str, retrieved_contexts: List[str]) -> float:
        """
        Computes lexical and key-term density of the query inside retrieved chunks.
        """
        if not retrieved_contexts:
            return 0.0
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        scores = []
        for ctx in retrieved_contexts:
            ctx_lower = ctx.lower()
            matches = sum(1 for term in query_terms if term in ctx_lower)
            scores.append(matches / float(len(query_terms)))

        return round(sum(scores) / len(scores), 4)

    def evaluate_query(
        self,
        query: str,
        retrieved_ids: List[str],
        retrieved_contexts: List[str],
        expected_doc_ids: List[str],
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-metric evaluation on a single query retrieval.
        """
        hit_rate = self.calculate_hit_rate(retrieved_ids[:k], expected_doc_ids)
        mrr = self.calculate_mrr(retrieved_ids, expected_doc_ids)
        precision_at_k = self.calculate_precision_at_k(retrieved_ids, expected_doc_ids, k)
        relevance = self.calculate_context_relevance(query, retrieved_contexts)

        return {
            "query": query,
            "evaluated_k": k,
            "metrics": {
                "hit_rate_at_k": hit_rate,
                "mean_reciprocal_rank": mrr,
                "precision_at_k": precision_at_k,
                "context_relevance_score": relevance
            },
            "status": "EXCELLENT" if mrr >= 0.8 else "ACCEPTABLE" if mrr >= 0.5 else "POOR"
        }


def evaluate_retrieval_accuracy(
    query: str,
    retrieved_ids: List[str],
    retrieved_contexts: List[str],
    expected_doc_ids: List[str],
    k: int = 3
) -> Dict[str, Any]:
    evaluator = RetrievalEvaluator()
    return evaluator.evaluate_query(query, retrieved_ids, retrieved_contexts, expected_doc_ids, k)
