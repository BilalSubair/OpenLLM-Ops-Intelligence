"""
Docker & Containerization Specialist Agent.
Diagnoses NVIDIA Container Toolkit, GPU passthrough flags,
shared memory (/dev/shm) Bus errors, and Docker daemon issues.
"""

from typing import Dict, Any, Optional
from copilot.agents.base_agent import BaseSpecialistAgent
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.config_inspector import ConfigInspector
from copilot.tools.searxng_client import SearXNGClient


class DockerInfrastructureAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Docker-Infrastructure-Agent",
            specialty="Docker Container Runtime, NVIDIA Container Toolkit, GPU Passthrough, and /dev/shm",
            system_prompt="You are a senior Container Infrastructure Specialist focusing on NVIDIA Container Toolkit and Docker AI runtimes."
        )
        self.log_analyzer = LogAnalyzer()
        self.config_inspector = ConfigInspector()
        self.search_client = SearXNGClient()

    async def diagnose(self, problem_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.clear_steps()
        context = context or {}

        # Step 1: Inspect Docker Run command or Compose file if provided
        docker_config = context.get("dockerfile") or context.get("compose") or context.get("command")
        config_findings = None
        if docker_config:
            self.log_step(
                thought="Auditing Docker deployment configuration for GPU passthrough and shared memory sizing.",
                action="ConfigInspector.inspect_docker_config"
            )
            config_findings = self.config_inspector.inspect_docker_config(docker_config)
            self.steps[-1].observation = f"Score: {config_findings['score']}/100, Issues: {config_findings['findings_count']}"

        # Step 2: Log analysis
        log_data = context.get("logs") or problem_description
        self.log_step(
            thought="Scanning container logs for Bus errors, missing GPU driver flags, or NVIDIA Container Toolkit failures.",
            action="LogAnalyzer.analyze_log"
        )
        log_findings = self.log_analyzer.analyze_log(log_data)
        self.steps[-1].observation = {
            "issues_count": log_findings["issues_found"],
            "primary": log_findings.get("primary_diagnosis", {}).get("error_type") if log_findings.get("primary_diagnosis") else None
        }

        # Step 3: Determine Root Cause
        primary = log_findings.get("primary_diagnosis")
        if primary and primary.get("error_type") == "DOCKER_SHM_EXHAUSTED":
            root_cause = "Shared Memory Exhaustion (/dev/shm): Default Docker container shared memory is 64MB, causing PyTorch/vLLM multi-threading Bus errors."
            severity = "HIGH"
            remediation = [
                "1. Add '--shm-size=16g' or '--ipc=host' to docker run.",
                "2. In docker-compose.yml, add 'shm_size: 16gb'.",
                "3. Ensure the host has adequate RAM to back the shared memory tmpfs mount."
            ]
        elif primary and primary.get("error_type") == "NVIDIA_CONTAINER_TOOLKIT_MISSING":
            root_cause = "NVIDIA Container Toolkit Missing or Misconfigured: Docker daemon cannot hook into host NVIDIA drivers."
            severity = "CRITICAL"
            remediation = [
                "1. Install nvidia-container-toolkit on host.",
                "2. Run 'sudo nvidia-ctk runtime configure --runtime=docker'.",
                "3. Run 'sudo systemctl restart docker'.",
                "4. Test with 'docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi'."
            ]
        else:
            root_cause = "Docker containerization or GPU passthrough configuration issue."
            severity = "MEDIUM"
            remediation = [
                "Ensure '--gpus all' is provided when starting LLM containers.",
                "Increase container shared memory to at least 8GB-16GB.",
                "Verify Docker daemon default-runtime is set to 'nvidia' in /etc/docker/daemon.json."
            ]

        return {
            "agent": self.name,
            "specialty": self.specialty,
            "root_cause_analysis": root_cause,
            "severity": severity,
            "docker_lint_results": config_findings,
            "detected_log_issues": log_findings.get("all_detected_issues", []),
            "remediation_steps": remediation,
            "reasoning_steps": [s.to_dict() for s in self.steps]
        }
