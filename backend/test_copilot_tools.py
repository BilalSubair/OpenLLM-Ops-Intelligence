import unittest
import asyncio
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.config_inspector import ConfigInspector
from copilot.tools.searxng_client import SearXNGClient


class TestCopilotToolsSuite(unittest.TestCase):
    def setUp(self):
        self.log_analyzer = LogAnalyzer()
        self.config_inspector = ConfigInspector()
        self.searxng = SearXNGClient()

    def test_log_analyzer_cuda_oom(self):
        sample_log = """
        [2026-11-10 10:15:02] Loading model weights into device: cuda:0
        [2026-11-10 10:15:05] Allocating KV cache with context length 8192
        RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 7.79 GiB total capacity; 6.10 GiB already allocated)
        """
        analysis = self.log_analyzer.analyze_log(sample_log)
        self.assertEqual(analysis["issues_found"], 1)
        self.assertIsNotNone(analysis["primary_diagnosis"])
        self.assertEqual(analysis["primary_diagnosis"]["error_type"], "CUDA_OUT_OF_MEMORY")
        self.assertEqual(analysis["primary_diagnosis"]["severity"], "CRITICAL")
        self.assertGreater(len(analysis["primary_diagnosis"]["remediation"]), 0)

    def test_log_analyzer_docker_bus_error(self):
        sample_log = """
        Epoch 1/10: Training batch 120
        RuntimeError: DataLoader worker (pid 412) is killed by signal: Bus error.
        It is possible that dataloader's workers are out of shared memory.
        """
        analysis = self.log_analyzer.analyze_log(sample_log)
        self.assertEqual(analysis["issues_found"], 1)
        self.assertEqual(analysis["primary_diagnosis"]["error_type"], "DOCKER_SHM_EXHAUSTED")

    def test_log_analyzer_k8s_oom_killed(self):
        sample_log = """
        Events:
          Type     Reason     Age   From               Message
          ----     ------     ----  ----               -------
          Warning  OOMKilled  2m    kubelet, node-gpu1 Container vllm terminated with exit code 137
        """
        analysis = self.log_analyzer.analyze_log(sample_log)
        self.assertGreaterEqual(analysis["issues_found"], 1)
        self.assertEqual(analysis["primary_diagnosis"]["error_type"], "K8S_OOM_KILLED")

    def test_config_inspector_k8s_manifest(self):
        # Flawed manifest missing GPU limit and /dev/shm
        bad_manifest = """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: vllm-inference
        spec:
          template:
            spec:
              containers:
                - name: vllm
                  image: vllm/vllm-openai:latest
        """
        result = self.config_inspector.inspect_kubernetes_manifest(bad_manifest)
        self.assertEqual(result["status"], "FAILED")
        self.assertLess(result["score"], 80)
        rules = [f["rule"] for f in result["findings"]]
        self.assertIn("MISSING_GPU_RESOURCE_LIMIT", rules)
        self.assertIn("MISSING_SHARED_MEMORY_VOLUME", rules)

    def test_config_inspector_docker_config(self):
        bad_docker = "docker run -d -p 8000:8000 vllm/vllm-openai:latest"
        result = self.config_inspector.inspect_docker_config(bad_docker)
        self.assertEqual(result["status"], "FAILED")
        rules = [f["rule"] for f in result["findings"]]
        self.assertIn("GPU_PASSTHROUGH_DISABLED", rules)
        self.assertIn("DEFAULT_SHM_SIZE_RISK", rules)

    def test_searxng_fallback(self):
        res = asyncio.run(self.searxng.search("CUDA out of memory fix"))
        self.assertIn("results", res)
        self.assertGreater(res["results_count"], 0)


if __name__ == "__main__":
    unittest.main()
