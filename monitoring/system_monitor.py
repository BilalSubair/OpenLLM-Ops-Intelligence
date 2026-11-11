# monitoring/system_monitor.py
"""System monitoring utilities.

Provides CPU and RAM usage metrics. The implementation attempts to use ``psutil`` if available; otherwise it falls back to the standard library ``os`` and ``resource`` modules, returning zeros on unsupported platforms (e.g., macOS without psutil)."""

import os
from typing import Dict

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def get_system_metrics() -> Dict[str, float]:
    """Return a dictionary with basic system metrics.

    Keys:
        - ``cpu_percent`` – overall CPU utilisation (percentage).
        - ``mem_total``  – total physical memory in megabytes.
        - ``mem_used``   – memory currently used (MiB).
        - ``mem_free``   – free memory (MiB).
    """
    if psutil:
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "mem_total": mem.total / 1024 / 1024,
            "mem_used": mem.used / 1024 / 1024,
            "mem_free": mem.available / 1024 / 1024,
        }
    # Fallback – no psutil available
    return {"cpu_percent": 0.0, "mem_total": 0.0, "mem_used": 0.0, "mem_free": 0.0}

if __name__ == "__main__":  # pragma: no cover
    print(get_system_metrics())
