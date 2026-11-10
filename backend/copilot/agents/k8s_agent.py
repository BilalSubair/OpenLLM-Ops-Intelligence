"""
Kubernetes AI Infrastructure Specialist Agent.
Diagnoses GPU Pod scheduling, OOMKilled events, CrashLoopBackOff,
taints/tolerations, and NVIDIA GPU Operator issues.
"""

from typing import Dict, Any, Optional
from copilot.agents.base_agent import BaseSpecialistAgent
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.config_inspector import ConfigInspector
from copilot.tools.searxng_client import SearXNGClient


class KubernetesInfrastructureAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Kubernetes-Infrastructure-Agent",
            specialty="Kubernetes GPU Workloads, Scheduling, Taints/Tolerations, and GPU Operator",
            system_prompt="You are a senior Kubernetes Platform Engineer specializing in ML Platform and GPU workload orchestration."
        )
        self.log_analyzer = LogAnalyzer()
        self.config_inspector = ConfigInspector()
        self.search_client = SearXNGClient()

    async def diagnose(self, problem_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.clear_steps()
        context = context or {}

        # Step 1: Inspect Manifest if YAML was provided
        manifest_data = context.get("manifest") or context.get("yaml")
        manifest_findings = None
        if manifest_data:
            self.log_step(
                thought="Auditing Kubernetes workload manifest against GPU scheduling and memory best practices.",
                action="ConfigInspector.inspect_kubernetes_manifest"
            )
            manifest_findings = self.config_inspector.inspect_kubernetes_manifest(manifest_data)
            self.steps[-1].observation = f"Score: {manifest_findings['score']}/100, Issues: {manifest_findings['findings_count']}"

        # Step 2: Log analysis
        log_data = context.get("logs") or problem_description
        self.log_step(
            thought="Parsing Kubernetes event logs for OOMKilled (ExitCode 137) or FailedScheduling events.",
            action="LogAnalyzer.analyze_log"
        )
        log_findings = self.log_analyzer.analyze_log(log_data)
        self.steps[-1].observation = {
            "issues_count": log_findings["issues_found"],
            "primary": log_findings.get("primary_diagnosis", {}).get("error_type") if log_findings.get("primary_diagnosis") else None
        }

        # Step 3: Determine Root Cause
        primary = log_findings.get("primary_diagnosis")
        if primary and primary.get("error_type") == "K8S_OOM_KILLED":
            root_cause = "Container Terminated by Linux OOM Killer (ExitCode 137): Host memory limit exceeded."
            severity = "CRITICAL"
            remediation = [
                "1. Increase container memory limit in pod spec: resources.limits.memory.",
                "2. Check if PyTorch CPU worker threads or KV cache host memory is leaking.",
                "3. Configure requests closer to limits to avoid burst evictions."
            ]
        elif primary and primary.get("error_type") == "K8S_INSUFFICIENT_GPU":
            root_cause = "Pod Scheduling Failure: No available nodes with requested GPU capacity or matching taints."
            severity = "HIGH"
            remediation = [
                "1. Check if NVIDIA GPU Operator daemonsets are running in 'gpu-operator' namespace.",
                "2. Add matching toleration: key: nvidia.com/gpu, operator: Exists, effect: NoSchedule.",
                "3. Check cluster allocatable GPU resources: 'kubectl describe nodes | grep nvidia.com/gpu'."
            ]
        elif manifest_findings and manifest_findings["findings"]:
            root_cause = f"Manifest Configuration Issue: {manifest_findings['findings'][0]['message']}"
            severity = manifest_findings["findings"][0]["severity"]
            remediation = [f["fix"] for f in manifest_findings["findings"]]
        else:
            root_cause = "Kubernetes GPU pod configuration or scheduling failure."
            severity = "MEDIUM"
            remediation = [
                "Verify 'nvidia-device-plugin' is running healthy on cluster nodes.",
                "Ensure pod has 'resources.limits.nvidia.com/gpu: 1' defined.",
                "Check 'kubectl describe pod <name>' for scheduler failure details."
            ]

        return {
            "agent": self.name,
            "specialty": self.specialty,
            "root_cause_analysis": root_cause,
            "severity": severity,
            "manifest_lint_results": manifest_findings,
            "detected_log_issues": log_findings.get("all_detected_issues", []),
            "remediation_steps": remediation,
            "reasoning_steps": [s.to_dict() for s in self.steps]
        }
