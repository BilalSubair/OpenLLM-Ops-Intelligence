"""
GGUF Quantization Trade-Off & Performance Analyzer.
Focuses on evaluating trade-offs between model size, VRAM footprint,
token throughput, and perplexity degradation for GGUF formats,
with special emphasis on Q4_K_M vs Q8_0.
"""

from typing import Dict, Any, List, Optional


class QuantizationAnalyzer:
    """
    Quantization specification and empirical analytical formulas
    derived from llama.cpp benchmarks and GGUF quantization specifications.
    """

    # Format specifications: (bits_per_weight, avg_ppl_increase, quantization_speed_factor)
    GGUF_FORMATS = {
        "FP16": {"bpw": 16.0, "ppl_delta": 0.0000, "overhead_factor": 1.00, "quality_grade": "Reference"},
        "Q8_0": {"bpw": 8.5, "ppl_delta": 0.0004, "overhead_factor": 0.98, "quality_grade": "Near Lossless"},
        "Q6_K": {"bpw": 6.56, "ppl_delta": 0.0035, "overhead_factor": 0.95, "quality_grade": "Excellent"},
        "Q5_K_M": {"bpw": 5.50, "ppl_delta": 0.0150, "overhead_factor": 0.92, "quality_grade": "Very Good"},
        "Q4_K_M": {"bpw": 4.50, "ppl_delta": 0.0530, "overhead_factor": 0.88, "quality_grade": "Good (Recommended Sweetspot)"},
        "Q3_K_M": {"bpw": 3.50, "ppl_delta": 0.2400, "overhead_factor": 0.82, "quality_grade": "Moderate Loss"},
        "Q2_K": {"bpw": 2.63, "ppl_delta": 0.8600, "overhead_factor": 0.74, "quality_grade": "Significant Degradation"}
    }

    # Reference GPU memory bandwidths in GB/s
    GPU_BANDWIDTH = {
        "NVIDIA_RTX_4090": 1008.0,
        "NVIDIA_A100_80GB": 2039.0,
        "NVIDIA_H100_SXM": 3350.0,
        "NVIDIA_L4": 300.0,
        "NVIDIA_RTX_3060": 360.0,
        "APPLE_M3_MAX": 400.0,
        "APPLE_M2_PRO": 200.0,
        "DEFAULT": 800.0
    }

    @staticmethod
    def calculate_model_size_gb(params_billion: float, bpw: float) -> float:
        """Calculate model file/weight size in gigabytes."""
        bytes_total = (params_billion * 1e9 * bpw) / 8.0
        return round(bytes_total / (1024 ** 3), 2)

    @staticmethod
    def calculate_kv_cache_gb(
        params_billion: float,
        context_length: int = 4096,
        layers: int = 32,
        kv_heads: int = 8,
        head_dim: int = 128,
        precision_bytes: int = 2
    ) -> float:
        """KV cache memory = 2 * layers * kv_heads * head_dim * context_length * precision_bytes."""
        kv_bytes = 2 * layers * kv_heads * head_dim * context_length * precision_bytes
        return round(kv_bytes / (1024 ** 3), 2)

    def analyze_quantization(
        self,
        params_billion: float = 8.0,
        context_length: int = 4096,
        gpu_bandwidth_gb_s: float = 1008.0
    ) -> Dict[str, Any]:
        """
        Calculates trade-offs across all standard GGUF quantization formats.
        """
        results = {}
        kv_cache_gb = self.calculate_kv_cache_gb(params_billion, context_length)
        cuda_overhead_gb = 0.8  # standard CUDA driver / PyTorch context overhead

        for fmt, specs in self.GGUF_FORMATS.items():
            bpw = specs["bpw"]
            weight_size_gb = self.calculate_model_size_gb(params_billion, bpw)
            total_vram_gb = round(weight_size_gb + kv_cache_gb + cuda_overhead_gb, 2)

            # In auto-regressive generation, speed is largely memory-bandwidth bound:
            # max_tokens_per_sec = Memory_Bandwidth / Weight_Size
            theoretical_tps = round(gpu_bandwidth_gb_s / max(weight_size_gb, 0.1), 1)

            results[fmt] = {
                "format": fmt,
                "bits_per_weight": bpw,
                "model_size_gb": weight_size_gb,
                "kv_cache_gb": kv_cache_gb,
                "total_vram_required_gb": total_vram_gb,
                "theoretical_max_tokens_sec": theoretical_tps,
                "estimated_ppl_delta": specs["ppl_delta"],
                "quality_grade": specs["quality_grade"],
                "fits_in_8gb_gpu": total_vram_gb <= 8.0,
                "fits_in_16gb_gpu": total_vram_gb <= 16.0,
                "fits_in_24gb_gpu": total_vram_gb <= 24.0
            }

        return {
            "parameters_billion": params_billion,
            "context_length": context_length,
            "gpu_bandwidth_gb_s": gpu_bandwidth_gb_s,
            "kv_cache_footprint_gb": kv_cache_gb,
            "formats": results
        }

    def compare_q4_vs_q8(
        self,
        params_billion: float = 8.0,
        context_length: int = 4096,
        gpu_bandwidth_gb_s: float = 1008.0
    ) -> Dict[str, Any]:
        """
        Specific comparative analysis between Q4_K_M and Q8_0 as emphasized in project requirements.
        """
        q4 = self.GGUF_FORMATS["Q4_K_M"]
        q8 = self.GGUF_FORMATS["Q8_0"]

        kv_cache_gb = self.calculate_kv_cache_gb(params_billion, context_length)
        overhead = 0.8

        q4_size = self.calculate_model_size_gb(params_billion, q4["bpw"])
        q8_size = self.calculate_model_size_gb(params_billion, q8["bpw"])

        q4_vram = round(q4_size + kv_cache_gb + overhead, 2)
        q8_vram = round(q8_size + kv_cache_gb + overhead, 2)

        q4_tps = round(gpu_bandwidth_gb_s / max(q4_size, 0.1), 1)
        q8_tps = round(gpu_bandwidth_gb_s / max(q8_size, 0.1), 1)

        vram_savings_gb = round(q8_vram - q4_vram, 2)
        vram_savings_pct = round(((q8_vram - q4_vram) / q8_vram) * 100, 1)
        speedup_factor = round(q4_tps / max(q8_tps, 0.1), 2)

        return {
            "comparison": "Q4_K_M vs Q8_0",
            "model_parameters": f"{params_billion}B",
            "context_length": context_length,
            "metrics": {
                "Q4_K_M": {
                    "model_size_gb": q4_size,
                    "total_vram_gb": q4_vram,
                    "theoretical_tokens_sec": q4_tps,
                    "perplexity_increase": q4["ppl_delta"],
                    "quality_grade": q4["quality_grade"]
                },
                "Q8_0": {
                    "model_size_gb": q8_size,
                    "total_vram_gb": q8_vram,
                    "theoretical_tokens_sec": q8_tps,
                    "perplexity_increase": q8["ppl_delta"],
                    "quality_grade": q8["quality_grade"]
                }
            },
            "trade_off_analysis": {
                "vram_savings_gb": vram_savings_gb,
                "vram_savings_percent": vram_savings_pct,
                "throughput_speedup": f"{speedup_factor}x faster with Q4_K_M",
                "quality_trade_off": "Q8_0 offers negligible quality loss (+0.0004 PPL), while Q4_K_M saves ~47% VRAM with modest perplexity impact (+0.053 PPL).",
                "recommendation": (
                    "Deploy Q4_K_M for high-throughput serving, consumer GPUs (8GB/16GB), or edge deployments. "
                    "Deploy Q8_0 for precision-critical reasoning, code generation, or when ample VRAM (>=24GB) is available."
                )
            }
        }


def analyze_quantization_tradeoffs(
    params_billion: float = 8.0,
    context_length: int = 4096,
    gpu_type: str = "NVIDIA_RTX_4090"
) -> Dict[str, Any]:
    analyzer = QuantizationAnalyzer()
    bandwidth = analyzer.GPU_BANDWIDTH.get(gpu_type, analyzer.GPU_BANDWIDTH["DEFAULT"])
    general = analyzer.analyze_quantization(params_billion, context_length, bandwidth)
    q4_vs_q8 = analyzer.compare_q4_vs_q8(params_billion, context_length, bandwidth)
    return {
        "analysis": general,
        "head_to_head_q4_vs_q8": q4_vs_q8
    }
