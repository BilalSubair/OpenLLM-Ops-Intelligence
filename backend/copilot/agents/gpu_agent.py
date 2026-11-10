"""
GPU & CUDA Infrastructure Specialist Agent.
Diagnoses GPU out-of-memory (OOM), CUDA kernel errors, thermal throttling,
and multi-GPU NCCL synchronization faults.
"""

from typing import Dict, Any, Optional
from copilot.agents.base_agent import BaseSpecialistAgent
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.searxng_client import SearXNGClient
from monitoring.gpu_monitor import get_gpu_metrics


class GPUInfrastructureAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="GPU-Infrastructure-Agent",
            specialty="GPU Hardware, VRAM Allocation, CUDA Kernels, and NCCL",
            system_prompt="You are a senior GPU Systems Engineer specializing in NVIDIA CUDA, VRAM optimization, and multi-GPU clusters."
        )
        self.log_analyzer = LogAnalyzer()
        self.search_client = SearXNGClient()

    async def diagnose(self, problem_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.clear_steps()
        context = context or {}

        # Step 1: Reason and check current live GPU hardware state
        self.log_step(
            thought="Inspecting live GPU hardware telemetry to check current VRAM allocation and device status.",
            action="get_gpu_metrics"
        )
        gpu_telemetry = get_gpu_metrics()
        self.steps[-1].observation = {
            "device_count": gpu_telemetry.get("device_count", 0),
            "vram_utilization_percent": gpu_telemetry.get("vram_utilization_percent", 0.0),
            "total_vram_mb": gpu_telemetry.get("total_vram_mb", 0.0)
        }

        # Step 2: Log analysis if logs are provided in context or problem
        log_data = context.get("logs") or problem_description
        self.log_step(
            thought="Executing pattern recognition across provided log traces to identify CUDA or Xid crash signatures.",
            action="LogAnalyzer.analyze_log"
        )
        log_findings = self.log_analyzer.analyze_log(log_data)
        self.steps[-1].observation = {
            "issues_found": log_findings["issues_found"],
            "primary": log_findings.get("primary_diagnosis", {}).get("error_type") if log_findings.get("primary_diagnosis") else None
        }

        # Step 3: Web search / knowledge augmentation if required
        search_results = []
        if log_findings.get("primary_diagnosis"):
            err_type = log_findings["primary_diagnosis"]["error_type"]
            self.log_step(
                thought=f"Querying knowledge repository via SearXNG for verified resolutions to {err_type}.",
                action="SearXNGClient.search"
            )
            search_data = await self.search_client.search(f"NVIDIA CUDA error {err_type} fix")
            search_results = search_data.get("results", [])
            self.steps[-1].observation = f"Retrieved {len(search_results)} relevant technical articles."

        # Step 4: Synthesize root cause analysis and actionable remediation
        is_oom = "out of memory" in problem_description.lower() or "oom" in problem_description.lower() or (
            log_findings.get("primary_diagnosis", {}).get("error_type") == "CUDA_OUT_OF_MEMORY"
        )

        if is_oom:
            root_cause = "GPU Memory (VRAM) Exhaustion: Model weights + KV cache + intermediate activations exceeded physical memory capacity."
            severity = "CRITICAL"
            remediation = [
                "1. Quantization: Transition model to GGUF Q4_K_M or AWQ 4-bit to cut model weights by ~50-70%.",
                "2. Context Truncation: Reduce max_model_len or max context window length.",
                "3. KV Cache Paging: Enable vLLM PagedAttention or set --gpu-memory-utilization 0.90.",
                "4. Batch Tuning: Lower max concurrent request batch size.",
                "5. Tensor Parallelism: If multiple GPUs are available, shard across GPUs with --tensor-parallel-size."
            ]
        elif log_findings.get("primary_diagnosis", {}).get("error_type") == "NVIDIA_XID_ERROR":
            root_cause = "Hardware / Kernel Driver Fault: NVIDIA driver reported an Xid error event."
            severity = "FATAL"
            remediation = log_findings["primary_diagnosis"]["remediation"]
        else:
            root_cause = "General GPU performance degradation or allocation latency."
            severity = "MEDIUM"
            remediation = [
                "Check for background processes locking GPU VRAM using 'nvidia-smi'.",
                "Ensure PyTorch CUDA allocator is not suffering from memory fragmentation: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
            ]

        return {
            "agent": self.name,
            "specialty": self.specialty,
            "root_cause_analysis": root_cause,
            "severity": severity,
            "gpu_telemetry_snapshot": gpu_telemetry,
            "detected_log_issues": log_findings.get("all_detected_issues", []),
            "remediation_steps": remediation,
            "reference_sources": search_results[:2],
            "reasoning_steps": [s.to_dict() for s in self.steps]
        }
