"""
Inference Service Health Monitor.
Performs continuous and on-demand health probes across inference backends.
"""

import time
from typing import Dict, Any, List
from benchmarking.engines import EngineRegistry


class EngineHealthMonitor:
    def __init__(self):
        self.engine_registry = EngineRegistry()
        self.cached_status: Dict[str, Any] = {}
        self.last_check_time: float = 0.0

    async def probe_all(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Probes all registered engines and returns live connectivity status and latencies.
        """
        now = time.time()
        if not force_refresh and (now - self.last_check_time < 5.0) and self.cached_status:
            return self.cached_status

        probes = await self.engine_registry.get_all_health()

        online_count = sum(1 for p in probes.values() if p.get("status") == "online")
        total_count = len(probes)

        self.cached_status = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "online_engines": online_count,
            "total_engines": total_count,
            "system_health": "OPTIMAL" if online_count >= 2 else "DEGRADED" if online_count == 1 else "OFFLINE",
            "engines": probes
        }
        self.last_check_time = now
        return self.cached_status


# Global health monitor instance
health_monitor = EngineHealthMonitor()
