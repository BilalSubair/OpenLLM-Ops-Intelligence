import unittest
import asyncio
from copilot.agents.gpu_agent import GPUInfrastructureAgent
from copilot.agents.k8s_agent import KubernetesInfrastructureAgent
from copilot.agents.docker_agent import DockerInfrastructureAgent
from copilot.agents.rag_diagnostics_agent import RAGDiagnosticsAgent
from copilot.orchestrator import copilot_orchestrator


class TestAgentsSuite(unittest.TestCase):
    def test_gpu_agent_oom_diagnosis(self):
        agent = GPUInfrastructureAgent()
        diag = asyncio.run(agent.diagnose("CUDA out of memory while loading Llama-3-70B on RTX 4090"))
        self.assertEqual(diag["agent"], "GPU-Infrastructure-Agent")
        self.assertEqual(diag["severity"], "CRITICAL")
        self.assertIn("GPU Memory", diag["root_cause_analysis"])
        self.assertGreater(len(diag["remediation_steps"]), 0)
        self.assertGreater(len(diag["reasoning_steps"]), 0)

    def test_k8s_agent_diagnosis(self):
        agent = KubernetesInfrastructureAgent()
        diag = asyncio.run(agent.diagnose(
            "Pod is failing to schedule with 0/4 nodes available: Insufficient nvidia.com/gpu and untolerated taint"
        ))
        self.assertEqual(diag["agent"], "Kubernetes-Infrastructure-Agent")
        self.assertEqual(diag["severity"], "HIGH")
        self.assertIn("Pod Scheduling Failure", diag["root_cause_analysis"])
        self.assertGreater(len(diag["remediation_steps"]), 0)

    def test_docker_agent_diagnosis(self):
        agent = DockerInfrastructureAgent()
        diag = asyncio.run(agent.diagnose(
            "PyTorch DataLoader worker killed by Bus error inside container",
            context={"command": "docker run -p 8000:8000 my-image"}
        ))
        self.assertEqual(diag["agent"], "Docker-Infrastructure-Agent")
        self.assertIn("Shared Memory", diag["root_cause_analysis"])

    def test_orchestrator_routing_and_triage(self):
        assigned = copilot_orchestrator.route_incident("GPU ran out of VRAM in Kubernetes pod")
        self.assertIn("gpu", assigned)
        self.assertIn("kubernetes", assigned)

        triage = asyncio.run(copilot_orchestrator.troubleshoot(
            "GPU out of memory in Kubernetes container"
        ))
        self.assertEqual(triage["status"], "diagnosed")
        self.assertIn("assigned_agents", triage["coordination"])
        self.assertIn("root_cause_analysis", triage)
        self.assertIn("consolidated_remediation", triage)


if __name__ == "__main__":
    unittest.main()
