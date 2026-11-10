"""
Automated Benchmark Runner.
Executes concurrent or sequential latency and throughput benchmarks
across Ollama, vLLM, llama.cpp, TGI, and LMDeploy.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from benchmarking.engines import EngineRegistry, BaseInferenceEngine
from monitoring.inference_metrics import inference_metrics_collector


class BenchmarkRunner:
    def __init__(self):
        self.registry = EngineRegistry()

    async def benchmark_engine(
        self,
        engine_name: str,
        prompt: str = "Explain the cause of CUDA out of memory errors in distributed LLM training.",
        model: Optional[str] = None,
        iterations: int = 3,
        max_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        Run a multi-iteration benchmark on a single inference engine.
        """
        engine = self.registry.get_engine(engine_name)
        if not engine:
            return {"error": f"Unknown engine: {engine_name}"}

        health = await engine.check_health()
        if health.get("status") != "online":
            return {
                "engine": engine_name,
                "status": "offline",
                "health_details": health,
                "message": f"Engine {engine_name} is unreachable at {engine.base_url}"
            }

        latencies = []
        tokens_per_sec_list = []
        completion_tokens_list = []
        samples = []

        for i in range(iterations):
            try:
                res = await engine.generate(prompt=prompt, model=model, max_tokens=max_tokens)
                latencies.append(res["total_time_ms"])
                tokens_per_sec_list.append(res["tokens_per_second"])
                completion_tokens_list.append(res["completion_tokens"])
                samples.append({
                    "iteration": i + 1,
                    "latency_ms": res["total_time_ms"],
                    "tokens_per_second": res["tokens_per_second"],
                    "tokens_generated": res["completion_tokens"]
                })

                # Register in global telemetry
                inference_metrics_collector.record_request(
                    model=res.get("model", "unknown"),
                    engine=engine_name,
                    latency_ms=res["total_time_ms"],
                    prompt_tokens=res.get("prompt_tokens", 0),
                    completion_tokens=res["completion_tokens"],
                    ttft_ms=res.get("ttft_ms", 0.0),
                    success=True
                )
            except Exception as e:
                samples.append({
                    "iteration": i + 1,
                    "error": str(e)
                })

        valid_latencies = [l for l in latencies if l > 0]
        if not valid_latencies:
            return {
                "engine": engine_name,
                "status": "failed",
                "samples": samples,
                "message": "All benchmark iterations failed"
            }

        avg_lat = round(sum(valid_latencies) / len(valid_latencies), 2)
        min_lat = round(min(valid_latencies), 2)
        max_lat = round(max(valid_latencies), 2)
        avg_tps = round(sum(tokens_per_sec_list) / len(tokens_per_sec_list), 2) if tokens_per_sec_list else 0.0

        return {
            "engine": engine_name,
            "status": "success",
            "iterations_completed": len(valid_latencies),
            "benchmark_summary": {
                "avg_latency_ms": avg_lat,
                "min_latency_ms": min_lat,
                "max_latency_ms": max_lat,
                "avg_tokens_per_sec": avg_tps,
                "avg_tokens_generated": round(sum(completion_tokens_list) / len(completion_tokens_list), 1)
            },
            "samples": samples
        }

    async def compare_engines(
        self,
        engines: List[str],
        prompt: str = "Troubleshoot Kubernetes CrashLoopBackOff for a GPU inference pod.",
        iterations: int = 2,
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Runs benchmarks in parallel or sequence across multiple engines for comparison.
        """
        comparison_results = {}
        for eng in engines:
            comparison_results[eng] = await self.benchmark_engine(
                engine_name=eng,
                prompt=prompt,
                iterations=iterations,
                max_tokens=max_tokens
            )

        # Leaderboard ranking based on avg_tokens_per_sec
        ranked = []
        for eng, res in comparison_results.items():
            if res.get("status") == "success":
                tps = res["benchmark_summary"]["avg_tokens_per_sec"]
                lat = res["benchmark_summary"]["avg_latency_ms"]
                ranked.append({"engine": eng, "avg_tokens_per_sec": tps, "avg_latency_ms": lat})

        ranked.sort(key=lambda x: x["avg_tokens_per_sec"], reverse=True)

        return {
            "prompt": prompt,
            "engines_evaluated": engines,
            "leaderboard": ranked,
            "detailed_results": comparison_results
        }
