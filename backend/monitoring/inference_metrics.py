"""
Inference Performance & Token Telemetry Collector.
Tracks latency, throughput, token generation speed (tokens/sec), and time-to-first-token (TTFT).
"""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import deque


class InferenceMetricsCollector:
    def __init__(self, window_size: int = 500):
        self.lock = threading.Lock()
        self.window_size = window_size
        self.total_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.start_time = time.time()

        # Deques for rolling window calculations
        self.latencies: deque = deque(maxlen=window_size)
        self.ttfts: deque = deque(maxlen=window_size)
        self.tokens_per_sec: deque = deque(maxlen=window_size)
        self.recent_timestamps: deque = deque(maxlen=window_size)

        # Engine and model breakdowns
        self.engine_counts: Dict[str, int] = {}
        self.model_counts: Dict[str, int] = {}

    def record_request(
        self,
        model: str,
        engine: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_ms: Optional[float] = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """
        Record completed LLM inference invocation and calculate immediate metrics.
        """
        now = time.time()
        with self.lock:
            self.total_requests += 1
            if not success:
                self.failed_requests += 1

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            self.latencies.append(latency_ms)
            self.recent_timestamps.append(now)

            if ttft_ms is not None:
                self.ttfts.append(ttft_ms)

            # Calculate generation speed (tokens/sec)
            tps = 0.0
            if completion_tokens > 0 and latency_ms > 0:
                gen_time_sec = (latency_ms - (ttft_ms or 0)) / 1000.0
                if gen_time_sec <= 0:
                    gen_time_sec = latency_ms / 1000.0
                tps = round(completion_tokens / max(gen_time_sec, 0.001), 2)
                self.tokens_per_sec.append(tps)

            self.engine_counts[engine] = self.engine_counts.get(engine, 0) + 1
            self.model_counts[model] = self.model_counts.get(model, 0) + 1

        return {
            "model": model,
            "engine": engine,
            "latency_ms": round(latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "tokens_per_second": tps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Computes aggregate statistics, percentiles, throughput (req/s), and token speeds.
        """
        with self.lock:
            elapsed_sec = max(1.0, time.time() - self.start_time)
            success_count = self.total_requests - self.failed_requests

            # Calculate windowed throughput (req/sec in last 60 seconds)
            now = time.time()
            cutoff = now - 60.0
            recent_count = sum(1 for t in self.recent_timestamps if t >= cutoff)
            throughput_60s = round(recent_count / 60.0, 2)

            # Percentiles for latency
            lat_list = sorted(list(self.latencies)) if self.latencies else [0.0]
            count = len(lat_list)

            def get_percentile(p: float) -> float:
                if not lat_list:
                    return 0.0
                idx = int(p * count)
                idx = min(idx, count - 1)
                return round(lat_list[idx], 2)

            avg_latency = round(sum(lat_list) / count, 2) if count > 0 else 0.0
            p50_latency = get_percentile(0.50)
            p95_latency = get_percentile(0.95)
            p99_latency = get_percentile(0.99)

            # TTFT statistics
            ttft_list = list(self.ttfts)
            avg_ttft = round(sum(ttft_list) / len(ttft_list), 2) if ttft_list else 0.0

            # Tokens/sec statistics
            tps_list = list(self.tokens_per_sec)
            avg_tps = round(sum(tps_list) / len(tps_list), 2) if tps_list else 0.0
            peak_tps = round(max(tps_list), 2) if tps_list else 0.0

            error_rate = round((self.failed_requests / max(1, self.total_requests)) * 100, 2)

            return {
                "uptime_seconds": round(elapsed_sec, 1),
                "total_requests": self.total_requests,
                "successful_requests": success_count,
                "failed_requests": self.failed_requests,
                "error_rate_percent": error_rate,
                "throughput_requests_per_sec": throughput_60s,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "latency_ms": {
                    "avg": avg_latency,
                    "p50": p50_latency,
                    "p95": p95_latency,
                    "p99": p99_latency
                },
                "time_to_first_token_ms": {
                    "avg": avg_ttft,
                    "sample_count": len(ttft_list)
                },
                "token_generation_speed": {
                    "avg_tokens_per_sec": avg_tps,
                    "peak_tokens_per_sec": peak_tps
                },
                "breakdown_by_engine": dict(self.engine_counts),
                "breakdown_by_model": dict(self.model_counts)
            }


# Singleton collector
inference_metrics_collector = InferenceMetricsCollector()
