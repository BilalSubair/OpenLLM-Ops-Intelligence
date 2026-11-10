"""
Model Fallback & Intelligent Router.
Provides automated failover routing across inference engines (e.g., vLLM -> Ollama -> llama.cpp)
to ensure high availability for LLM infrastructure.
"""

import time
from typing import Dict, Any, List, Optional
from benchmarking.engines import EngineRegistry
from registry.model_registry import model_registry
from registry.audit_logger import audit_logger
from monitoring.inference_metrics import inference_metrics_collector


class FallbackRouter:
    def __init__(self):
        self.engine_registry = EngineRegistry()
        # Fallback hierarchy preference order
        self.fallback_priority = ["vllm", "ollama", "llama.cpp", "tgi", "lmdeploy"]

    async def route_and_generate(
        self,
        prompt: str,
        preferred_engine: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        Attempts inference on the preferred engine. If unreachable, gracefully
        fails over to secondary engines in order of priority.
        """
        reg_model = model_registry.get_model(model_id) if model_id else None
        primary_engine_name = preferred_engine or (reg_model.preferred_engine if reg_model else "ollama")

        # Build execution order: primary first, followed by others in fallback hierarchy
        execution_chain = [primary_engine_name]
        for eng in self.fallback_priority:
            if eng != primary_engine_name and eng not in execution_chain:
                execution_chain.append(eng)

        attempt_trace = []
        last_error = None

        for engine_name in execution_chain:
            engine = self.engine_registry.get_engine(engine_name)
            if not engine:
                continue

            health = await engine.check_health()
            if health.get("status") != "online":
                attempt_trace.append({
                    "engine": engine_name,
                    "status": "skipped",
                    "reason": "Health check offline"
                })
                continue

            try:
                start_time = time.perf_counter()
                result = await engine.generate(
                    prompt=prompt,
                    model=model_id or engine.default_model,
                    max_tokens=max_tokens
                )
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                is_failover = (engine_name != primary_engine_name)

                attempt_trace.append({
                    "engine": engine_name,
                    "status": "success",
                    "latency_ms": latency_ms
                })

                # Register in telemetry & audit log
                inference_metrics_collector.record_request(
                    model=model_id or engine.default_model,
                    engine=engine_name,
                    latency_ms=latency_ms,
                    prompt_tokens=result.get("prompt_tokens", 0),
                    completion_tokens=result.get("completion_tokens", 0),
                    ttft_ms=result.get("ttft_ms", 0.0),
                    success=True
                )

                audit_logger.log_event(
                    event_type="INFERENCE_COMPLETED",
                    model=model_id or engine.default_model,
                    engine=engine_name,
                    latency_ms=latency_ms,
                    status="SUCCESS",
                    metadata={"failover_active": is_failover, "attempts": attempt_trace}
                )

                return {
                    "text": result.get("text", ""),
                    "engine_used": engine_name,
                    "model_used": model_id or engine.default_model,
                    "failover_triggered": is_failover,
                    "attempts": attempt_trace,
                    "metrics": {
                        "latency_ms": latency_ms,
                        "tokens_per_second": result.get("tokens_per_second", 0.0),
                        "tokens_generated": result.get("completion_tokens", 0)
                    }
                }
            except Exception as e:
                last_error = str(e)
                attempt_trace.append({
                    "engine": engine_name,
                    "status": "error",
                    "error": last_error
                })

        # If all backends failed
        audit_logger.log_event(
            event_type="INFERENCE_FAILED",
            model=model_id or "unknown",
            engine=primary_engine_name,
            latency_ms=0.0,
            status="FAILED_ALL_ENGINES",
            metadata={"attempts": attempt_trace}
        )

        return {
            "error": "All inference backends in fallback chain failed or were unreachable",
            "last_error": last_error,
            "attempts": attempt_trace
        }


# Global fallback router instance
fallback_router = FallbackRouter()
