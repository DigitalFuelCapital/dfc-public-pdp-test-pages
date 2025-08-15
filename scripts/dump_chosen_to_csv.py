#!/usr/bin/env python3
"""
Dump run_index, shopper_name, test_name, model, datetime, chosen_url and justification from a JSONL run log to CSV.

Default input: data/output/raw_runs.jsonl
Default output: data/output/chosen_justifications.csv

Usage:
  python scripts/dump_chosen_to_csv.py
  python scripts/dump_chosen_to_csv.py --input data/output/raw_runs.jsonl --output data/output/chosen_justifications.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _extract_fields(rec: Dict[str, Any]) -> Tuple[Any, str, str, str, str, str, str]:
    """
    Safely extract (run_index, shopper_name, test_name, model, datetime, chosen_url, justification) from one JSON record.
    Returns empty strings for missing values.
    """
    run_index = rec.get("run_index", "")
    shopper_name = rec.get("shopper_name") or ""
    test_name = rec.get("test_name") or ""
    model = rec.get("model") or ""
    
    # Convert timestamp to readable datetime
    timestamp = rec.get("timestamp", "")
    datetime_str = ""
    if timestamp:
        try:
            dt = datetime.fromtimestamp(int(timestamp))
            datetime_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            datetime_str = str(timestamp)
    
    result = rec.get("result") or {}

    chosen_url = ""
    justification = ""

    if isinstance(result, dict):
        chosen_url = result.get("chosen_url") or ""
        justification = result.get("justification") or ""
    # If result is not a dict (rare), leave as empty strings.

    # Normalize whitespace a bit on justification
    if isinstance(justification, str):
        justification = " ".join(justification.split())

    return run_index, str(shopper_name), str(test_name), str(model), datetime_str, str(chosen_url), justification


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Generator reading JSONL file, yielding dict per valid JSON line.
    Skips empty lines and logs parse issues to stderr-like prints.
    """
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping line {total}: JSON parse error: {e}")
                continue
            if not isinstance(obj, dict):
                print(f"[warn] Skipping line {total}: expected object, got {type(obj).__name__}")
                continue
            yield obj


def write_csv(rows: Iterable[Tuple[Any, str, str, str, str, str, str]], out_path: Path) -> None:
    """
    Write rows to CSV with header: run_index,shopper_name,test_name,model,datetime,chosen_url,justification
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_index", "shopper_name", "test_name", "model", "datetime", "chosen_url", "justification"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump run_index, shopper_name, test_name, model, datetime, chosen_url and justification from raw_runs.jsonl to CSV.")
    parser.add_argument(
        "--input",
        "-i",
        default="data/output/raw_runs.jsonl",
        help="Path to input JSONL file (default: data/output/raw_runs.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/output/chosen_justifications.csv",
        help="Path to output CSV file (default: data/output/chosen_justifications.csv)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    records = list(read_jsonl(in_path))
    rows = [_extract_fields(rec) for rec in records]
    write_csv(rows, out_path)

    total = len(records)
    written = len(rows)
    print(f"[ok] Processed {total} JSONL records")
    print(f"[ok] Wrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()