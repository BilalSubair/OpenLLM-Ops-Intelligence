import unittest
from rag_advanced.evaluation import RetrievalEvaluator, evaluate_retrieval_accuracy
from rag_advanced.hallucination import HallucinationDetector, assess_hallucination


class TestRAGAdvancedSuite(unittest.TestCase):
    def test_retrieval_evaluation_metrics(self):
        evaluator = RetrievalEvaluator()

        retrieved = ["doc_chunk_1", "doc_chunk_2", "doc_chunk_3", "doc_chunk_4"]
        ground_truth = ["doc_chunk_2"]

        # Hit rate@1 should be 0, Hit rate@2 should be 1.0
        hr1 = evaluator.calculate_hit_rate(retrieved[:1], ground_truth)
        hr2 = evaluator.calculate_hit_rate(retrieved[:2], ground_truth)
        self.assertEqual(hr1, 0.0)
        self.assertEqual(hr2, 1.0)

        # MRR: doc_chunk_2 is at rank 2 => 1/2 = 0.5
        mrr = evaluator.calculate_mrr(retrieved, ground_truth)
        self.assertEqual(mrr, 0.5)

        # Context relevance
        query = "GPU memory allocation OOM"
        contexts = [
            "This guide covers GPU memory allocation and handling OOM crashes.",
            "Kubernetes network ingress routing rules."
        ]
        relevance = evaluator.calculate_context_relevance(query, contexts)
        self.assertGreater(relevance, 0.0)

        # Comprehensive evaluation wrapper
        summary = evaluate_retrieval_accuracy(
            query=query,
            retrieved_ids=retrieved,
            retrieved_contexts=contexts,
            expected_doc_ids=ground_truth,
            k=3
        )
        self.assertEqual(summary["metrics"]["mean_reciprocal_rank"], 0.5)
        self.assertEqual(summary["metrics"]["hit_rate_at_k"], 1.0)

    def test_hallucination_detector(self):
        detector = HallucinationDetector()

        context = """
        NVIDIA H100 SXM5 GPUs provide 80GB of HBM3 memory with a bandwidth of 3.35 TB/s.
        The power consumption of an H100 SXM module is approximately 700 Watts.
        """

        # Grounded answer
        grounded_answer = "The NVIDIA H100 SXM5 features 80GB of HBM3 memory and consumes up to 700 Watts."
        res1 = detector.assess_faithfulness(context, grounded_answer)
        self.assertGreaterEqual(res1["faithfulness_percent"], 80.0)
        self.assertLessEqual(res1["hallucination_score"], 0.20)
        self.assertEqual(res1["status"], "GROUNDED")

        # Hallucinated answer
        hallucinated_answer = "Quantum supercomputers utilize superconducting transmon qubits cooled with liquid helium to calculate cosmological models."
        res2 = detector.assess_faithfulness(context, hallucinated_answer)
        self.assertLessEqual(res2["faithfulness_percent"], 40.0)
        self.assertGreaterEqual(res2["hallucination_score"], 0.60)
        self.assertEqual(res2["status"], "HALLUCINATION_DETECTED")


if __name__ == "__main__":
    unittest.main()
