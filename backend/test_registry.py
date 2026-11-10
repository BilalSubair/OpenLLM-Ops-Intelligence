import unittest
import asyncio
from registry.model_registry import model_registry, RegisteredModel
from registry.health_monitor import health_monitor
from registry.fallback_router import fallback_router
from registry.audit_logger import audit_logger


class TestRegistrySuite(unittest.TestCase):
    def test_model_registry(self):
        models = model_registry.list_models()
        self.assertGreater(len(models), 0)
        qwen = model_registry.get_model("qwen3.5:2b")
        self.assertIsNotNone(qwen)
        self.assertEqual(qwen.quantization, "Q4_K_M")

        # Register custom model
        custom = RegisteredModel(
            model_id="test-model-1",
            display_name="Test Model 1",
            parameters_billion=7.0,
            quantization="Q8_0",
            preferred_engine="vllm",
            context_window=4096,
            vram_minimum_gb=10.0,
            recommended_hardware="RTX 4090"
        )
        model_registry.register_model(custom)
        fetched = model_registry.get_model("test-model-1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.quantization, "Q8_0")

    def test_health_monitor(self):
        status = asyncio.run(health_monitor.probe_all(force_refresh=True))
        self.assertIn("timestamp", status)
        self.assertIn("engines", status)
        self.assertIn("ollama", status["engines"])
        self.assertIn("vllm", status["engines"])

    def test_audit_logger(self):
        entry = audit_logger.log_event(
            event_type="TEST_EVENT",
            model="test-model",
            engine="ollama",
            latency_ms=45.5,
            status="SUCCESS",
            metadata={"test_key": "test_val"}
        )
        self.assertEqual(entry["event_type"], "TEST_EVENT")
        recent = audit_logger.get_recent_logs(limit=10)
        self.assertTrue(any(e["audit_id"] == entry["audit_id"] for e in recent))


if __name__ == "__main__":
    unittest.main()
