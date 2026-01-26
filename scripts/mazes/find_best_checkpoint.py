#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


def main() -> int:
    metrics = Path("logs/metrics.csv")
    if not metrics.exists():
        return 1

    best_score = float("-inf")
    best_id = None
    with metrics.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model_id = (row.get("model_id") or "").strip()
            if not model_id:
                continue
            try:
                score = float(row.get("avg_reward", float("-inf")))
            except Exception:
                score = float("-inf")
            if score > best_score:
                best_score = score
                best_id = model_id

    if not best_id:
        return 1

    candidate = Path("models") / f"{best_id}_latest.zip"
    if not candidate.exists():
        return 1

    print(candidate.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
