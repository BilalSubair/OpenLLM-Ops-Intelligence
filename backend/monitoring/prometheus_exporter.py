"""
Prometheus Metrics Exporter Module.
Formats GPU, host system, and inference telemetry into standard Prometheus exposition format.
"""

from typing import Dict, Any
from monitoring.gpu_monitor import get_gpu_metrics
from monitoring.system_monitor import get_system_metrics
from monitoring.inference_metrics import inference_metrics_collector


def generate_prometheus_metrics() -> str:
    """
    Generates OpenMetrics / Prometheus compatible text format.
    """
    gpu = get_gpu_metrics()
    sys_m = get_system_metrics()
    inf = inference_metrics_collector.get_summary()

    lines = []

    # HELP and TYPE annotations
    lines.append("# HELP llm_ops_gpu_utilization_percent Overall GPU compute utilization percentage")
    lines.append("# TYPE llm_ops_gpu_utilization_percent gauge")
    lines.append(f"llm_ops_gpu_utilization_percent {gpu.get('overall_gpu_utilization_percent', 0.0)}")

    lines.append("# HELP llm_ops_vram_total_bytes Total VRAM in bytes")
    lines.append("# TYPE llm_ops_vram_total_bytes gauge")
    lines.append(f"llm_ops_gpu_vram_total_bytes {gpu.get('total_vram_mb', 0.0) * 1024 * 1024}")

    lines.append("# HELP llm_ops_vram_used_bytes Currently allocated VRAM in bytes")
    lines.append("# TYPE llm_ops_vram_used_bytes gauge")
    lines.append(f"llm_ops_gpu_vram_used_bytes {gpu.get('used_vram_mb', 0.0) * 1024 * 1024}")

    lines.append("# HELP llm_ops_vram_utilization_percent Percentage of VRAM utilized")
    lines.append("# TYPE llm_ops_vram_utilization_percent gauge")
    lines.append(f"llm_ops_gpu_vram_utilization_percent {gpu.get('vram_utilization_percent', 0.0)}")

    for dev in gpu.get("devices", []):
        d_id = dev.get("device_id", 0)
        d_name = dev.get("name", "unknown").replace('"', '')
        lines.append(f'llm_ops_device_gpu_utilization{{device_id="{d_id}",name="{d_name}"}} {dev.get("gpu_utilization_percent", 0.0)}')
        lines.append(f'llm_ops_device_vram_used_bytes{{device_id="{d_id}",name="{d_name}"}} {dev.get("vram_used_mb", 0.0) * 1024 * 1024}')
        lines.append(f'llm_ops_device_temperature_celsius{{device_id="{d_id}",name="{d_name}"}} {dev.get("temperature_c", 0.0)}')
        lines.append(f'llm_ops_device_power_watts{{device_id="{d_id}",name="{d_name}"}} {dev.get("power_draw_w", 0.0)}')

    # System metrics
    lines.append("# HELP llm_ops_host_cpu_utilization_percent Host CPU usage percentage")
    lines.append("# TYPE llm_ops_host_cpu_utilization_percent gauge")
    lines.append(f"llm_ops_host_cpu_utilization_percent {sys_m['cpu']['utilization_percent']}")

    lines.append("# HELP llm_ops_host_memory_utilization_percent Host RAM usage percentage")
    lines.append("# TYPE llm_ops_host_memory_utilization_percent gauge")
    lines.append(f"llm_ops_host_memory_utilization_percent {sys_m['memory']['utilization_percent']}")

    # Inference metrics
    lines.append("# HELP llm_ops_inference_requests_total Total number of inference requests")
    lines.append("# TYPE llm_ops_inference_requests_total counter")
    lines.append(f"llm_ops_inference_requests_total {inf['total_requests']}")

    lines.append("# HELP llm_ops_inference_failed_requests_total Total number of failed inference requests")
    lines.append("# TYPE llm_ops_inference_failed_requests_total counter")
    lines.append(f"llm_ops_inference_failed_requests_total {inf['failed_requests']}")

    lines.append("# HELP llm_ops_inference_latency_avg_ms Average inference latency in ms")
    lines.append("# TYPE llm_ops_inference_latency_avg_ms gauge")
    lines.append(f"llm_ops_inference_latency_avg_ms {inf['latency_ms']['avg']}")

    lines.append("# HELP llm_ops_inference_latency_p95_ms 95th percentile inference latency in ms")
    lines.append("# TYPE llm_ops_inference_latency_p95_ms gauge")
    lines.append(f"llm_ops_inference_latency_p95_ms {inf['latency_ms']['p95']}")

    lines.append("# HELP llm_ops_inference_tokens_per_second Token generation speed in tokens per second")
    lines.append("# TYPE llm_ops_inference_tokens_per_second gauge")
    lines.append(f"llm_ops_inference_tokens_per_second {inf['token_generation_speed']['avg_tokens_per_sec']}")

    lines.append("# HELP llm_ops_inference_ttft_ms Average time to first token in ms")
    lines.append("# TYPE llm_ops_inference_ttft_ms gauge")
    lines.append(f"llm_ops_inference_ttft_ms {inf['time_to_first_token_ms']['avg']}")

    lines.append("")  # Trailing newline required by Prometheus
    return "\n".join(lines)
