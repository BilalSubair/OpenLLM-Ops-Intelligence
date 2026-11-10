"""
Automated Infrastructure Log Analyzer.
Detects, classifies, and pinpoints root causes in CUDA, GPU driver,
Docker, Kubernetes, and RAG service logs.
"""

import re
from typing import Dict, Any, List, Optional


class LogAnalyzer:
    """
    Pattern-matching heuristics and root-cause classifier for AI infrastructure logs.
    """

    PATTERNS = [
        # CUDA OOM
        {
            "category": "GPU_CUDA",
            "error_type": "CUDA_OUT_OF_MEMORY",
            "regex": r"(CUDA\s+out\s+of\s+memory|torch\.cuda\.OutOfMemoryError|RuntimeError:\s*CUDA\s+error:\s*out\s+of\s+memory|tried\s+to\s+allocate\s+([0-9\.]+)\s*(GiB|MiB))",
            "severity": "CRITICAL",
            "summary": "GPU VRAM exhausted during model loading, KV cache allocation, or batch execution.",
            "remediation": [
                "Reduce per-device batch size or max context window length.",
                "Enable 4-bit / 8-bit quantization (GGUF Q4_K_M or AWQ/GPTQ).",
                "Enable FlashAttention or PagedAttention (vLLM --gpu-memory-utilization 0.90).",
                "Offload layers to CPU/RAM or increase tensor parallelism (TP)."
            ]
        },
        # Driver / Xid Errors
        {
            "category": "GPU_DRIVER",
            "error_type": "NVIDIA_XID_ERROR",
            "regex": r"(NVRM:\s*Xid\s*\(PCI:[^\)]+\):\s*(\d+)|GPU\s+has\s+fallen\s+off\s+the\s+bus|CUDA\s+driver\s+version\s+is\s+insufficient)",
            "severity": "FATAL",
            "summary": "NVIDIA hardware or kernel driver crash (Xid event) or CUDA driver mismatch.",
            "remediation": [
                "Inspect dmesg / syslog for exact Xid code (e.g., Xid 31 = memory page fault, Xid 79 = GPU fallen off bus).",
                "Check GPU physical seating and PCIe auxiliary power connectors.",
                "Verify NVIDIA kernel module version matches user-space driver (nvidia-smi).",
                "Reboot host or reinstall NVIDIA datacenter driver."
            ]
        },
        # Docker Shared Memory OOM
        {
            "category": "DOCKER",
            "error_type": "DOCKER_SHM_EXHAUSTED",
            "regex": r"(Bus\s+error|RuntimeError:\s*DataLoader\s+worker|no\s+space\s+left\s+on\s+device\s*:\s*'?/dev/shm'?)",
            "severity": "HIGH",
            "summary": "Docker container ran out of shared memory (/dev/shm). Default is only 64MB.",
            "remediation": [
                "Pass '--shm-size=16g' or '--ipc=host' to docker run.",
                "In docker-compose.yml, add 'shm_size: 16gb'.",
                "In Kubernetes, mount an emptyDir with medium: Memory to /dev/shm."
            ]
        },
        # Docker GPU runtime missing
        {
            "category": "DOCKER",
            "error_type": "NVIDIA_CONTAINER_TOOLKIT_MISSING",
            "regex": r"(could\s+not\s+select\s+device\s+driver\s+\"\"?\s+with\s+capabilities:\s*\[\[gpu\]\]|unknown\s+flag:\s*--gpus)",
            "severity": "CRITICAL",
            "summary": "Docker daemon cannot access NVIDIA GPUs because NVIDIA Container Toolkit is not installed or configured as default runtime.",
            "remediation": [
                "Install nvidia-container-toolkit via apt/yum.",
                "Run 'sudo nvidia-ctk runtime configure --runtime=docker'.",
                "Restart Docker daemon: 'sudo systemctl restart docker'."
            ]
        },
        # Kubernetes OOMKilled
        {
            "category": "KUBERNETES",
            "error_type": "K8S_OOM_KILLED",
            "regex": r"(OOMKilled|ExitCode:\s*137|Terminated\s+with\s+exit\s+code\s+137|Killed\s+process\s+\d+\s*\([^\)]+\)\s*total-vm)",
            "severity": "CRITICAL",
            "summary": "Pod process exceeded its container memory limit and was terminated by the Linux kernel OOM killer (Exit Code 137).",
            "remediation": [
                "Increase 'resources.limits.memory' in the Pod / Deployment specification.",
                "Check for memory leaks or unbounded KV cache growth.",
                "Set memory requests closer to actual working set to assist the K8s scheduler."
            ]
        },
        # Kubernetes GPU Unschedulable
        {
            "category": "KUBERNETES",
            "error_type": "K8S_INSUFFICIENT_GPU",
            "regex": r"(0/\d+\s+nodes\s+(are\s+)?available|Insufficient\s+nvidia\.com/gpu|did\s+not\s+match\s+Pod\'s\s+node\s+selector/affinity|node\(s\)\s+had\s+untolerated\s+taint)",
            "severity": "HIGH",
            "summary": "Kubernetes scheduler cannot find a node matching GPU resource requests or node taints.",
            "remediation": [
                "Verify NVIDIA GPU Operator is running and nodes are advertising 'nvidia.com/gpu' allocatable resources.",
                "Add matching tolerations for GPU node taints (e.g., 'nvidia.com/gpu:NoSchedule').",
                "Ensure node labels match the required nodeSelector (e.g., accelerator: nvidia-h100)."
            ]
        },
        # RAG / Service Connection Error
        {
            "category": "RAG_PIPELINE",
            "error_type": "OLLAMA_CONNECTION_REFUSED",
            "regex": r"(Connection\s+refused|Cannot\s+connect\s+to\s+Ollama|httpx\.ConnectError|Failed\s+to\s+connect\s+to\s+localhost:11434)",
            "severity": "HIGH",
            "summary": "FastAPI backend cannot communicate with local Ollama service.",
            "remediation": [
                "Verify Ollama is running ('ollama serve' or 'systemctl status ollama').",
                "Check if port 11434 is listening: 'curl http://localhost:11434/api/tags'.",
                "Ensure environment variable OLLAMA_HOST=0.0.0.0:11434 if calling across Docker containers."
            ]
        }
    ]

    def analyze_log(self, log_content: str) -> Dict[str, Any]:
        """
        Scan log text and match against known operational issue patterns.
        """
        detected_issues = []
        lines = log_content.splitlines()

        for pattern in self.PATTERNS:
            regex = re.compile(pattern["regex"], re.IGNORECASE)
            matched_lines = []
            for i, line in enumerate(lines):
                if regex.search(line):
                    matched_lines.append({
                        "line_number": i + 1,
                        "content": line.strip()
                    })

            if matched_lines:
                detected_issues.append({
                    "category": pattern["category"],
                    "error_type": pattern["error_type"],
                    "severity": pattern["severity"],
                    "summary": pattern["summary"],
                    "occurrences": len(matched_lines),
                    "evidence": matched_lines[:5],  # Top 5 matching snippets
                    "remediation": pattern["remediation"]
                })

        # Calculate primary diagnostic outcome
        primary_issue = None
        if detected_issues:
            # Sort by severity (FATAL > CRITICAL > HIGH)
            severity_order = {"FATAL": 4, "CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}
            detected_issues.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
            primary_issue = detected_issues[0]

        return {
            "total_lines_analyzed": len(lines),
            "issues_found": len(detected_issues),
            "primary_diagnosis": primary_issue,
            "all_detected_issues": detected_issues
        }
