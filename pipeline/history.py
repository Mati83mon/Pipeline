"""Append-only generation log with bounded size and thread-safe writes.

Fixes over the original: the log is rotated instead of growing without limit,
reads use a bounded tail instead of ``readlines()`` over the whole file, a
corrupt line no longer takes down the History tab, and concurrent generations
cannot interleave partial JSON lines.
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class GenerationLogger:
    def __init__(self, log_file: str, max_entries: int = 2000) -> None:
        self.log_file = log_file
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(parent or ".", exist_ok=True)

    # -- writing -----------------------------------------------------------

    def log(
        self,
        gen_type: str,
        prompt: str,
        params: Dict[str, Any],
        duration: float,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
        model_id: Optional[str] = None,
        fallback: bool = False,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": gen_type,
            "model_id": model_id,
            "fallback": bool(fallback),
            "prompt": prompt,
            "params": params,
            "duration_seconds": round(float(duration), 2),
            "output_path": output_path,
            "error": error,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                return
            self._rotate_locked()

    def _rotate_locked(self) -> None:
        """Truncate the log to ``max_entries`` once it drifts 20% past the cap."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        if len(lines) <= self.max_entries * 1.2:
            return
        try:
            with open(self.log_file, "w", encoding="utf-8") as handle:
                handle.writelines(lines[-self.max_entries :])
        except OSError:
            pass

    # -- reading -----------------------------------------------------------

    def history(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Return the newest ``limit`` entries, most recent first."""
        limit = max(1, int(limit))
        if not os.path.exists(self.log_file):
            return []
        tail: deque[str] = deque(maxlen=limit)
        try:
            with open(self.log_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        tail.append(line)
        except OSError:
            return []

        entries: List[Dict[str, Any]] = []
        for line in tail:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written line must not break the UI
            if isinstance(parsed, dict):
                entries.append(parsed)
        entries.reverse()
        return entries
