"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the request details and begin latency tracking."""
        key = request_id or user_id
        started_at = datetime.now(timezone.utc)
        self._open[key] = started_at.timestamp()
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "input": text,
                "start_time": started_at.isoformat(),
            }
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching request with its result and latency."""
        key = request_id or user_id
        finished_at = datetime.now(timezone.utc)
        started_at = self._open.pop(key, finished_at.timestamp())
        latency_ms = max(0.0, (finished_at.timestamp() - started_at) * 1000)

        for entry in reversed(self.logs):
            if entry.get("request_id") == request_id and entry.get("user_id") == user_id:
                entry.update(
                    {
                        "output": text,
                        "blocked": blocked,
                        "layer": layer,
                        "latency_ms": latency_ms,
                        "end_time": finished_at.isoformat(),
                    }
                )
                break
        else:
            self.logs.append(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "output": text,
                    "blocked": blocked,
                    "layer": layer,
                    "latency_ms": latency_ms,
                    "end_time": finished_at.isoformat(),
                }
            )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
