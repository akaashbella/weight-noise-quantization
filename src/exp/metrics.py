"""Append metrics to jsonl and write one-off JSON metrics."""

from __future__ import annotations

import json
import os
import tempfile


def append_metrics_jsonl(path: str, record: dict) -> None:
    """Append one JSON line to file (one record per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_metrics_json(path: str, data: dict) -> None:
    """Write a single JSON file atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix="metrics.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
