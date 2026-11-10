import unittest
from monitoring.gpu_monitor import get_gpu_metrics
from monitoring.system_monitor import get_system_metrics
from monitoring.inference_metrics import InferenceMetricsCollector
from monitoring.prometheus_exporter import generate_prometheus_metrics


class TestMonitoringSuite(unittest.TestCase):
    def test_gpu_metrics_structure(self):
        metrics = get_gpu_metrics()
        self.assertIn("status", metrics)
        self.assertIn("provider", metrics)
        self.assertIn("overall_gpu_utilization_percent", metrics)
        self.assertIn("total_vram_mb", metrics)
        self.assertIn("devices", metrics)
        self.assertIsInstance(metrics["devices"], list)

    def test_system_metrics_structure(self):
        sys_m = get_system_metrics()
        self.assertIn("cpu", sys_m)
        self.assertIn("memory", sys_m)
        self.assertIn("disk", sys_m)
        self.assertIn("process", sys_m)
        self.assertGreater(sys_m["memory"]["total_mb"], 0)

    def test_inference_metrics_collector(self):
        collector = InferenceMetricsCollector()
        rec = collector.record_request(
            model="qwen3.5:2b",
            engine="ollama",
            latency_ms=120.0,
            prompt_tokens=50,
            completion_tokens=25,
            ttft_ms=20.0,
            success=True
        )
        self.assertEqual(rec["model"], "qwen3.5:2b")
        self.assertGreater(rec["tokens_per_second"], 0.0)

        summary = collector.get_summary()
        self.assertEqual(summary["total_requests"], 1)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertIn("latency_ms", summary)
        self.assertIn("token_generation_speed", summary)

    def test_prometheus_exporter(self):
        output = generate_prometheus_metrics()
        self.assertIn("llm_ops_gpu_utilization_percent", output)
        self.assertIn("llm_ops_host_cpu_utilization_percent", output)
        self.assertIn("llm_ops_inference_requests_total", output)


if __name__ == "__main__":
    unittest.main()
