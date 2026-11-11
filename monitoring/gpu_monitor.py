# monitoring/gpu_monitor.py
"""GPU monitoring utilities.

The implementation is deliberately lightweight and works on systems **without** an NVIDIA GPU – it simply returns
zeroed metrics. When a GPU is present (e.g., on an NVIDIA‑enabled host), the code attempts to use ``nvidia-smi``
via ``subprocess`` and parses the output.

The functions are imported by ``backend/main.py`` and used by the Prometheus exporter.
"""

import subprocess
import re
from typing import Dict


def _run_nvidia_smi() -> str:
    """Run ``nvidia-smi`` and return its stdout.

    If the command is unavailable (e.g., on a MacBook M2) we catch the exception and return an empty string.
    """
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        # No NVIDIA driver / command not found
        return ""


def _parse_nvidia_smi(output: str) -> Dict[str, float]:
    """Parse the CSV output from ``nvidia-smi``.

    Expected format per line::

        <utilization>, <total_mem>, <used_mem>, <free_mem>

    Returns a dict with float values; if parsing fails we fall back to zeros.
    """
    if not output:
        return {"gpu_util": 0.0, "mem_total": 0.0, "mem_used": 0.0, "mem_free": 0.0}
    try:
        # Only handle the first GPU – multi‑GPU can be added later.
        line = output.splitlines()[0]
        util, total, used, free = [float(v) for v in re.split(r",\s*", line)]
        return {"gpu_util": util, "mem_total": total, "mem_used": used, "mem_free": free}
    except Exception:
        return {"gpu_util": 0.0, "mem_total": 0.0, "mem_used": 0.0, "mem_free": 0.0}


def get_gpu_metrics() -> Dict[str, float]:
    """Public helper returning current GPU usage.

    The returned dictionary is compatible with Prometheus metric generation:

    .. code-block:: python

        {
            "gpu_util": 12.5,      # percent
            "mem_total": 24576.0,  # MiB
            "mem_used": 10240.0,   # MiB
            "mem_free": 14336.0,   # MiB
        }
    """
    raw = _run_nvidia_smi()
    return _parse_nvidia_smi(raw)

# If the module is executed directly, print a quick report – handy for debugging.
if __name__ == "__main__":
    print(get_gpu_metrics())
