"""
Infrastructure Configuration Inspector & Linter.
Audits Kubernetes manifests, Docker run commands, Docker Compose files,
and vLLM/Ollama launch configurations against AI infrastructure best practices.
"""

from typing import Dict, Any, List, Optional
import re


class ConfigInspector:
    """
    Automated configuration linter for AI and LLM workloads.
    """

    def inspect_kubernetes_manifest(self, manifest_text: str) -> Dict[str, Any]:
        """
        Validates Kubernetes YAML manifest against GPU scheduling best practices.
        """
        findings = []
        score = 100

        # Check for GPU resource limits
        has_gpu_limit = bool(re.search(r"nvidia\.com/gpu:\s*['\"]?\d+['\"]?", manifest_text))
        has_gpu_request = bool(re.search(r"resources:\s*.*requests:.*nvidia\.com/gpu", manifest_text, re.DOTALL))

        if not has_gpu_limit:
            findings.append({
                "severity": "CRITICAL",
                "rule": "MISSING_GPU_RESOURCE_LIMIT",
                "message": "No 'nvidia.com/gpu' limit specified. Kubernetes will schedule this pod onto non-GPU nodes.",
                "fix": "Add 'resources.limits.nvidia.com/gpu: 1' to container definition."
            })
            score -= 30

        # Check for matching limits and requests (Kubernetes Best Practice for GPUs)
        if has_gpu_limit and not has_gpu_request:
            findings.append({
                "severity": "INFO",
                "rule": "IMPLICIT_GPU_REQUEST",
                "message": "When limits.nvidia.com/gpu is defined, requests are automatically set to equal limits.",
                "fix": "Explicitly define matching requests for consistency."
            })

        # Check for shared memory volume mount (/dev/shm)
        has_shm = "/dev/shm" in manifest_text or "dshm" in manifest_text
        if not has_shm:
            findings.append({
                "severity": "HIGH",
                "rule": "MISSING_SHARED_MEMORY_VOLUME",
                "message": "Pod does not mount an emptyDir to /dev/shm. Multi-worker PyTorch/vLLM will crash with Bus errors.",
                "fix": "Mount an emptyDir with medium: Memory to /dev/shm."
            })
            score -= 25

        # Check for tolerations (GPU nodes usually have taints)
        has_tolerations = "tolerations:" in manifest_text
        if not has_tolerations:
            findings.append({
                "severity": "MEDIUM",
                "rule": "MISSING_GPU_NODE_TOLERATIONS",
                "message": "No tolerations detected. Dedicated GPU nodes tainted with 'nvidia.com/gpu:NoSchedule' will reject this pod.",
                "fix": "Add toleration: key: 'nvidia.com/gpu', operator: 'Exists', effect: 'NoSchedule'."
            })
            score -= 15

        # Check for liveness and readiness probes
        has_probes = "readinessProbe:" in manifest_text or "livenessProbe:" in manifest_text
        if not has_probes:
            findings.append({
                "severity": "LOW",
                "rule": "MISSING_HEALTH_PROBES",
                "message": "No liveness or readiness probes defined. Model warm-up might cause traffic to hit unready containers.",
                "fix": "Add readinessProbe with initialDelaySeconds set to model loading time (e.g. 60-120s)."
            })
            score -= 10

        return {
            "config_type": "Kubernetes Manifest",
            "score": max(0, score),
            "status": "PASSED" if score >= 80 else "NEEDS_IMPROVEMENT" if score >= 50 else "FAILED",
            "findings_count": len(findings),
            "findings": findings
        }

    def inspect_docker_config(self, config_text: str) -> Dict[str, Any]:
        """
        Inspects Docker run commands or Docker Compose configurations.
        """
        findings = []
        score = 100

        # Check for GPU passthrough
        has_gpus = "--gpus" in config_text or "deploy:" in config_text and "capabilities: [gpu]" in config_text or "nvidia" in config_text
        if not has_gpus:
            findings.append({
                "severity": "CRITICAL",
                "rule": "GPU_PASSTHROUGH_DISABLED",
                "message": "No '--gpus all' flag or NVIDIA device reservation in Compose. Container cannot access host CUDA hardware.",
                "fix": "Pass '--gpus all' in CLI or add device reservation in Compose."
            })
            score -= 40

        # Check for shared memory
        has_shm = "--shm-size" in config_text or "shm_size:" in config_text or "--ipc=host" in config_text
        if not has_shm:
            findings.append({
                "severity": "HIGH",
                "rule": "DEFAULT_SHM_SIZE_RISK",
                "message": "Default Docker shared memory size is 64MB. Large model tensors will cause Bus errors.",
                "fix": "Specify '--shm-size=16g' or '--ipc=host'."
            })
            score -= 30

        return {
            "config_type": "Docker / Compose",
            "score": max(0, score),
            "status": "PASSED" if score >= 80 else "FAILED",
            "findings_count": len(findings),
            "findings": findings
        }
