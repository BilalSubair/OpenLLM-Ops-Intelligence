"""
Audit Logging Module.
Records structured JSON audit trails for queries, routing events, latencies,
and system interventions for security, compliance, and post-mortem analysis.
"""

import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque


class AuditLogger:
    def __init__(self, log_dir: str = "data/logs", max_in_memory: int = 1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit.jsonl"
        self.in_memory_events: deque = deque(maxlen=max_in_memory)

    def log_event(
        self,
        event_type: str,
        model: str,
        engine: str,
        latency_ms: float,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        client_ip: str = "127.0.0.1"
    ) -> Dict[str, Any]:
        """
        Appends audit record to memory and JSONL persistent log.
        """
        record = {
            "audit_id": uuid.uuid4().hex,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch_timestamp": time.time(),
            "event_type": event_type,
            "model": model,
            "engine": engine,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "client_ip": client_ip,
            "metadata": metadata or {}
        }

        self.in_memory_events.append(record)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        return record

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Returns recent audit events in reverse chronological order.
        """
        return list(reversed(list(self.in_memory_events)))[:limit]


# Global audit logger instance
audit_logger = AuditLogger()
