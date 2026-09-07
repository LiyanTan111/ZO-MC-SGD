"""Lightweight CSV / JSON experiment logging."""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Optional


class RunLogger:
    """Append-only CSV logger; metadata sidecar JSON."""

    def __init__(self, run_dir: str, name: str = "run", meta: Optional[dict[str, Any]] = None):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.csv_path = os.path.join(run_dir, f"{name}.csv")
        self.meta_path = os.path.join(run_dir, f"{name}_meta.json")
        self._csv_file = None
        self._writer = None
        self._headers: Optional[list[str]] = None
        if meta is not None:
            with open(self.meta_path, "w") as fh:
                json.dump(meta, fh, indent=2, default=str)

    def log(self, row: dict[str, Any]) -> None:
        if self._csv_file is None:
            self._headers = list(row.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._headers)
            self._writer.writeheader()
        # only keep known keys
        self._writer.writerow({k: row.get(k, "") for k in self._headers})
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def timestamp_dir(base: str, tag: Optional[str] = None) -> str:
    """Make a timestamped subdir under base; returns its path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = ts if tag is None else f"{ts}_{tag}"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path
