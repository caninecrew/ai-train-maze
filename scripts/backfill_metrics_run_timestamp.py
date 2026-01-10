import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
METRICS_CSV = LOG_DIR / "metrics.csv"


def load_timestamp_map(log_dir: Path) -> dict:
    mapping = {}
    for report_path in sorted(log_dir.glob("run_report_*.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_ts = data.get("summary", {}).get("run_timestamp")
        cycles = data.get("summary", {}).get("cycles", [])
        if not run_ts or not isinstance(cycles, list):
            continue
        for cycle in cycles:
            ts = cycle.get("timestamp")
            if ts and ts not in mapping:
                mapping[ts] = run_ts
    return mapping


def load_run_timestamps(log_dir: Path) -> list[str]:
    runs = []
    for report_path in sorted(log_dir.glob("run_report_*.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_ts = data.get("summary", {}).get("run_timestamp")
        if run_ts:
            runs.append(run_ts)
    return sorted(set(runs))


def choose_run_timestamp(row_ts: str, run_timestamps: list[str]) -> str:
    if not run_timestamps:
        return ""
    candidates = [rt for rt in run_timestamps if rt <= row_ts]
    if candidates:
        return candidates[-1]
    return run_timestamps[0]


def backfill_metrics(metrics_csv: Path, timestamp_map: dict, run_timestamps: list[str]) -> int:
    if not metrics_csv.exists():
        print(f"Missing metrics file: {metrics_csv}")
        return 0
    rows = []
    updated = 0
    def _valid_ts(ts: str) -> bool:
        return len(ts) == 15 and ts[8] == "-" and ts.replace("-", "").isdigit()
    with metrics_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [f for f in (reader.fieldnames or []) if f]
        if "run_timestamp" not in fieldnames:
            fieldnames.append("run_timestamp")
        for row in reader:
            ts = (row.get("timestamp") or "").strip()
            if not row.get("run_timestamp"):
                if ts in timestamp_map:
                    row["run_timestamp"] = timestamp_map[ts]
                    updated += 1
                elif _valid_ts(ts):
                    chosen = choose_run_timestamp(ts, run_timestamps)
                    if chosen:
                        row["run_timestamp"] = chosen
                        updated += 1
            clean_row = {key: row.get(key, "") for key in fieldnames}
            rows.append(clean_row)
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return updated


def main() -> None:
    timestamp_map = load_timestamp_map(LOG_DIR)
    run_timestamps = load_run_timestamps(LOG_DIR)
    updated = backfill_metrics(METRICS_CSV, timestamp_map, run_timestamps)
    print(f"Backfill complete. Updated {updated} rows.")


if __name__ == "__main__":
    main()
