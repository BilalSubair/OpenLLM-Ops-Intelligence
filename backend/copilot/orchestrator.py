"""
Multi-Agent Copilot Orchestrator.
Coordinates specialized agents (GPU, Kubernetes, Docker, RAG Diagnostics),
handles agent routing, tool calling, multi-step reasoning, and consensus synthesis.
"""

from typing import Dict, Any, List, Optional
import time

from copilot.agents.gpu_agent import GPUInfrastructureAgent
from copilot.agents.k8s_agent import KubernetesInfrastructureAgent
from copilot.agents.docker_agent import DockerInfrastructureAgent
from copilot.agents.rag_diagnostics_agent import RAGDiagnosticsAgent
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.config_inspector import ConfigInspector
from copilot.tools.searxng_client import SearXNGClient


class CopilotOrchestrator:
    def __init__(self):
        self.gpu_agent = GPUInfrastructureAgent()
        self.k8s_agent = KubernetesInfrastructureAgent()
        self.docker_agent = DockerInfrastructureAgent()
        self.rag_agent = RAGDiagnosticsAgent()

        self.log_analyzer = LogAnalyzer()
        self.config_inspector = ConfigInspector()
        self.searxng = SearXNGClient()

    def route_incident(self, problem_description: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Determines which specialist agent(s) should be assigned based on incident keywords.
        """
        text = problem_description.lower()
        if context and context.get("logs"):
            text += " " + context["logs"].lower()

        assigned_agents = []

        # GPU keywords
        if any(w in text for w in ["gpu", "cuda", "vram", "xid", "nvidia", "out of memory", "oom", "temperature", "driver"]):
            assigned_agents.append("gpu")

        # Kubernetes keywords
        if any(w in text for w in ["k8s", "kubernetes", "pod", "node", "taint", "toleration", "crashloopbackoff", "operator"]):
            assigned_agents.append("kubernetes")

        # Docker keywords
        if any(w in text for w in ["docker", "container", "shm", "/dev/shm", "docker-compose", "container toolkit"]):
            assigned_agents.append("docker")

        # RAG keywords
        if any(w in text for w in ["rag", "chroma", "vector", "embedding", "retrieval", "bm25", "hallucination", "rerank"]):
            assigned_agents.append("rag")

        # Default fallback
        if not assigned_agents:
            assigned_agents.append("gpu")

        return assigned_agents

    async def troubleshoot(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-agent troubleshooting workflow with reasoning and remediation.
        """
        start_time = time.perf_counter()
        context = context or {}

        # 1. Route to specialists
        target_agents = self.route_incident(problem_description, context)
        agent_reports = {}

        for agent_key in target_agents:
            if agent_key == "gpu":
                agent_reports["gpu"] = await self.gpu_agent.diagnose(problem_description, context)
            elif agent_key == "kubernetes":
                agent_reports["kubernetes"] = await self.k8s_agent.diagnose(problem_description, context)
            elif agent_key == "docker":
                agent_reports["docker"] = await self.docker_agent.diagnose(problem_description, context)
            elif agent_key == "rag":
                agent_reports["rag"] = await self.rag_agent.diagnose(problem_description, context)

        # 2. Multi-Agent Synthesis & Consensus
        lead_agent_key = target_agents[0]
        lead_report = agent_reports[lead_agent_key]

        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Aggregate remediation steps across active agents
        all_remediations = []
        for rep in agent_reports.values():
            all_remediations.extend(rep.get("remediation_steps", []))

        return {
            "incident_query": problem_description,
            "status": "diagnosed",
            "coordination": {
                "assigned_agents": target_agents,
                "lead_agent": lead_report["agent"],
                "total_coordination_time_ms": total_time_ms
            },
            "root_cause_analysis": lead_report["root_cause_analysis"],
            "severity": lead_report["severity"],
            "consolidated_remediation": list(dict.fromkeys(all_remediations)),  # Remove duplicates preserving order
            "specialist_reports": agent_reports
        }


# Global orchestrator singleton
copilot_orchestrator = CopilotOrchestrator()
