import unittest
from benchmarking.quantization_analyzer import QuantizationAnalyzer, analyze_quantization_tradeoffs


class TestQuantizationSuite(unittest.TestCase):
    def setUp(self):
        self.analyzer = QuantizationAnalyzer()

    def test_model_size_calculations(self):
        # 8B model with 4.5 bpw (Q4_K_M)
        q4_size = self.analyzer.calculate_model_size_gb(params_billion=8.0, bpw=4.5)
        self.assertAlmostEqual(q4_size, 4.19, delta=0.2)

        # 8B model with 8.5 bpw (Q8_0)
        q8_size = self.analyzer.calculate_model_size_gb(params_billion=8.0, bpw=8.5)
        self.assertAlmostEqual(q8_size, 7.92, delta=0.3)
        self.assertGreater(q8_size, q4_size)

    def test_compare_q4_vs_q8(self):
        result = self.analyzer.compare_q4_vs_q8(params_billion=8.0, context_length=4096)
        self.assertEqual(result["comparison"], "Q4_K_M vs Q8_0")
        self.assertIn("Q4_K_M", result["metrics"])
        self.assertIn("Q8_0", result["metrics"])

        # Q4 should have smaller VRAM requirements
        q4_vram = result["metrics"]["Q4_K_M"]["total_vram_gb"]
        q8_vram = result["metrics"]["Q8_0"]["total_vram_gb"]
        self.assertLess(q4_vram, q8_vram)

        # Q4 should yield higher theoretical throughput
        q4_tps = result["metrics"]["Q4_K_M"]["theoretical_tokens_sec"]
        q8_tps = result["metrics"]["Q8_0"]["theoretical_tokens_sec"]
        self.assertGreater(q4_tps, q8_tps)

        # Tradeoff analysis details
        tradeoffs = result["trade_off_analysis"]
        self.assertIn("vram_savings_gb", tradeoffs)
        self.assertIn("throughput_speedup", tradeoffs)
        self.assertIn("recommendation", tradeoffs)

    def test_full_analysis_wrapper(self):
        res = analyze_quantization_tradeoffs(params_billion=7.0, context_length=2048)
        self.assertIn("analysis", res)
        self.assertIn("head_to_head_q4_vs_q8", res)
        self.assertIn("Q4_K_M", res["analysis"]["formats"])
        self.assertIn("Q8_0", res["analysis"]["formats"])


if __name__ == "__main__":
    unittest.main()
