"""Keep timestamp, Oil_temperature, Motor_current, and TP2; clean and split chronologically."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


SOURCE = Path(r"C:\Users\32828\Documents\GitHub\CSI4900\weekly-reports\week2\MetroPT3(AirCompressor).csv")
OUTPUT_DIR = Path(__file__).resolve().parent
KEEP = ("timestamp", "Oil_temperature", "Motor_current", "TP2")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def clean_row(raw: dict[str, str]) -> tuple[str, float, float, float] | None:
    """Return the selected validated fields, or None for a missing/invalid record."""
    try:
        timestamp = datetime.fromisoformat(raw["timestamp"].strip())
        oil_temperature = float(raw["Oil_temperature"])
        motor_current = float(raw["Motor_current"])
        tp2 = float(raw["TP2"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

    values = (oil_temperature, motor_current, tp2)
    if not all(math.isfinite(value) for value in values):
        return None
    return timestamp.strftime("%Y-%m-%d %H:%M:%S"), oil_temperature, motor_current, tp2


def scan() -> tuple[int, dict[str, int]]:
    """Validate the source and count clean chronological records before allocating splits."""
    statistics = {
        "source_rows": 0,
        "rows_dropped_missing_or_invalid": 0,
        "exact_adjacent_duplicate_rows_dropped": 0,
        "out_of_order_timestamps": 0,
    }
    valid_rows = 0
    prior_timestamp: str | None = None
    prior_record: tuple[str, float, float, float] | None = None

    with SOURCE.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        missing_columns = set(KEEP).difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing expected source columns: {sorted(missing_columns)}")

        for raw in reader:
            statistics["source_rows"] += 1
            record = clean_row(raw)
            if record is None:
                statistics["rows_dropped_missing_or_invalid"] += 1
                continue
            if prior_timestamp is not None and record[0] < prior_timestamp:
                statistics["out_of_order_timestamps"] += 1
            if record == prior_record:
                statistics["exact_adjacent_duplicate_rows_dropped"] += 1
                continue
            valid_rows += 1
            prior_timestamp = record[0]
            prior_record = record

    if statistics["out_of_order_timestamps"]:
        raise ValueError(
            "The source is not chronological. Refusing to make a temporal split without sorting first."
        )
    return valid_rows, statistics


def write_splits(valid_rows: int) -> dict[str, dict[str, str | int]]:
    """Write non-shuffled temporal splits without the source index or unused variables."""
    targets = {
        "train": int(valid_rows * SPLIT_RATIOS["train"]),
        "validation": int(valid_rows * SPLIT_RATIOS["validation"]),
    }
    targets["test"] = valid_rows - targets["train"] - targets["validation"]
    names = list(targets)
    cumulative_targets = {
        "train": targets["train"],
        "validation": targets["train"] + targets["validation"],
        "test": valid_rows,
    }

    summaries = {name: {"rows": 0, "start_timestamp": None, "end_timestamp": None} for name in names}
    output_files = {}
    writers = {}
    try:
        for name in names:
            file = (OUTPUT_DIR / f"{name}.csv").open("w", encoding="utf-8", newline="")
            output_files[name] = file
            writers[name] = csv.DictWriter(file, fieldnames=KEEP)
            writers[name].writeheader()

        split_index = 0
        written = 0
        prior_record: tuple[str, float, float, float] | None = None
        with SOURCE.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file)
            for raw in reader:
                record = clean_row(raw)
                if record is None or record == prior_record:
                    continue
                prior_record = record

                while (
                    split_index < len(names) - 1
                    and written >= cumulative_targets[names[split_index]]
                ):
                    split_index += 1
                name = names[split_index]
                timestamp, oil_temperature, motor_current, tp2 = record
                writers[name].writerow(
                    {
                        "timestamp": timestamp,
                        "Oil_temperature": f"{oil_temperature:.12g}",
                        "Motor_current": f"{motor_current:.12g}",
                        "TP2": f"{tp2:.12g}",
                    }
                )
                summaries[name]["rows"] += 1
                summaries[name]["start_timestamp"] = summaries[name]["start_timestamp"] or timestamp
                summaries[name]["end_timestamp"] = timestamp
                written += 1

        if written != valid_rows:
            raise RuntimeError(f"Expected {valid_rows} clean rows but wrote {written}.")
    finally:
        for file in output_files.values():
            file.close()
    return summaries


def main() -> None:
    valid_rows, statistics = scan()
    split_summaries = write_splits(valid_rows)
    summary = {
        "source": str(SOURCE),
        "retained_columns": list(KEEP),
        "cleaning_policy": "Removed source index/unused columns, invalid rows, and exact adjacent duplicates; retained physical units; no scaling applied.",
        "split_policy": "Chronological, non-shuffled 70% train / 15% validation / 15% test to avoid future-data leakage in this time-series dataset.",
        "statistics": statistics,
        "clean_rows": valid_rows,
        "splits": split_summaries,
    }
    (OUTPUT_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
