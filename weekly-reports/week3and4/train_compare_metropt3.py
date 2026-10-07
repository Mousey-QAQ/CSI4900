"""Run all four MetroPT-3 baseline models and write one Markdown result file.

Input files (beside this script): train.csv, validation.csv, test.csv.
Output file (beside this script): model_comparison.md.
The input CSVs are read only. No model files, charts, or result folders are made.

Install dependencies: python -m pip install numpy pandas scikit-learn
Run:                  python train_compare_metropt3.py

Only Oil_temperature, Motor_current, and TP2 are used as model inputs. The
timestamp is used to derive labels and make a chronological evaluation split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _restart_with_project_venv() -> None:
    """Use the repository environment when VS Code starts us with another Python."""
    project_root = Path(__file__).resolve().parents[2]
    project_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not project_python.is_file():
        return

    if Path(sys.executable).resolve() == project_python.resolve():
        return

    print(f"Using project Python environment: {project_python}", flush=True)
    completed = subprocess.run(
        [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(completed.returncode)


_restart_with_project_venv()

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SENSORS = ["Oil_temperature", "Motor_current", "TP2"]
CSV_COLUMNS = ["timestamp", *SENSORS]
RANDOM_SEED = 4900
MAX_NORMAL_TRAIN_ROWS = 120_000
TRAIN_END = pd.Timestamp("2020-06-01 00:00:00")
VALIDATION_END = pd.Timestamp("2020-07-01 00:00:00")

# Reported air-leak periods from UCI MetroPT-3 / week2 Data Description_Metro.pdf.
# Endpoints are inclusive. These are provisional row labels, not a separate
# sensor measurement or a directly supplied label column.
REPORTED_FAILURES = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


def load_and_label(data_dir: Path) -> pd.DataFrame:
    """Read the three cleaned files and mark the documented failure periods."""
    frames = []
    for name in ("train", "validation", "test"):
        path = data_dir / f"{name}.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if list(frame.columns) != CSV_COLUMNS:
            raise ValueError(f"{path} must contain exactly {CSV_COLUMNS}")
        if frame[CSV_COLUMNS].isna().any().any():
            raise ValueError(f"{path} contains a missing timestamp or sensor value")
        for sensor in SENSORS:
            frame[sensor] = pd.to_numeric(frame[sensor], errors="raise").astype("float32")
        if not np.isfinite(frame[SENSORS].to_numpy()).all():
            raise ValueError(f"{path} contains a non-finite sensor value")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("The three input CSVs must be in chronological order")
    data["failure_label"] = 0
    for start, end in REPORTED_FAILURES:
        data.loc[data["timestamp"].between(start, end), "failure_label"] = 1
    return data


def make_model_splits(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use time boundaries that leave a reported failure in each partition."""
    splits = {
        "train": data.loc[data["timestamp"] < TRAIN_END].copy(),
        "validation": data.loc[
            (data["timestamp"] >= TRAIN_END) & (data["timestamp"] < VALIDATION_END)
        ].copy(),
        "test": data.loc[data["timestamp"] >= VALIDATION_END].copy(),
    }
    for name, frame in splits.items():
        if frame.empty or frame["failure_label"].nunique() != 2:
            raise ValueError(f"{name} must contain normal and reported-failure rows")
    return splits


def choose_training_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Keep all reported failures and a repeatable normal subset for speed."""
    normal = train.loc[train["failure_label"] == 0]
    failure = train.loc[train["failure_label"] == 1]
    if len(normal) > MAX_NORMAL_TRAIN_ROWS:
        normal = normal.sample(n=MAX_NORMAL_TRAIN_ROWS, random_state=RANDOM_SEED)
    return pd.concat([normal, failure], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    )


def best_validation_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Maximize F1 on validation only; do not use test labels for tuning."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        raise ValueError("Validation scores have no usable threshold")
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    index = int(np.argmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def evaluate_supervised(name: str, model, sample: pd.DataFrame,
                        validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    model.fit(sample[SENSORS], sample["failure_label"])
    validation_scores = model.predict_proba(validation[SENSORS])[:, 1]
    threshold, validation_f1 = best_validation_threshold(
        validation["failure_label"].to_numpy(), validation_scores
    )
    test_scores = model.predict_proba(test[SENSORS])[:, 1]
    row = calculate_metrics(
        test["failure_label"].to_numpy(), (test_scores >= threshold).astype(int)
    )
    return {"model": name, **row, "validation_f1": validation_f1, "threshold": threshold}


# ---------------------------------------------------------------------------
# MODEL 1: LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
def run_logistic_regression(sample: pd.DataFrame, validation: pd.DataFrame,
                            test: pd.DataFrame) -> dict:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
    )
    return evaluate_supervised("Logistic Regression", model, sample, validation, test)


# ---------------------------------------------------------------------------
# MODEL 2: DECISION TREE
# ---------------------------------------------------------------------------
def run_decision_tree(sample: pd.DataFrame, validation: pd.DataFrame,
                      test: pd.DataFrame) -> dict:
    model = DecisionTreeClassifier(
        max_depth=9, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_SEED
    )
    return evaluate_supervised("Decision Tree", model, sample, validation, test)


# ---------------------------------------------------------------------------
# MODEL 3: RANDOM FOREST
# ---------------------------------------------------------------------------
def run_random_forest(sample: pd.DataFrame, validation: pd.DataFrame,
                      test: pd.DataFrame) -> dict:
    model = RandomForestClassifier(
        n_estimators=120, max_depth=12, min_samples_leaf=30,
        class_weight="balanced_subsample", n_jobs=4, random_state=RANDOM_SEED,
    )
    return evaluate_supervised("Random Forest", model, sample, validation, test)


# ---------------------------------------------------------------------------
# MODEL 4: ISOLATION FOREST
# ---------------------------------------------------------------------------
def run_isolation_forest(sample: pd.DataFrame, validation: pd.DataFrame,
                         test: pd.DataFrame) -> dict:
    # Isolation Forest learns normal behavior only. The scaler is also fitted
    # only to these normal training rows.
    normal = sample.loc[sample["failure_label"] == 0, SENSORS]
    scaler = StandardScaler().fit(normal)
    model = IsolationForest(
        n_estimators=100, max_samples=2048, n_jobs=4, random_state=RANDOM_SEED
    )
    model.fit(scaler.transform(normal))

    # Isolation Forest returns higher decision_function values for normal
    # observations, so reverse the sign to make larger = more anomalous.
    validation_scores = -model.decision_function(scaler.transform(validation[SENSORS]))
    threshold, validation_f1 = best_validation_threshold(
        validation["failure_label"].to_numpy(), validation_scores
    )
    test_scores = -model.decision_function(scaler.transform(test[SENSORS]))
    row = calculate_metrics(
        test["failure_label"].to_numpy(), (test_scores >= threshold).astype(int)
    )
    return {"model": "Isolation Forest", **row,
            "validation_f1": validation_f1, "threshold": threshold}


def markdown_table(table: pd.DataFrame, columns: list[str], as_percent: bool = False) -> str:
    """Format selected pandas DataFrame columns as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in table[columns].itertuples(index=False, name=None):
        cells = [str(values[0])]
        cells.extend(f"{value * 100:.2f}%" if as_percent else f"{int(value):,}" for value in values[1:])
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def make_report(results: pd.DataFrame) -> str:
    """Return only the six requested metrics for the test set."""
    test_results = results[
        ["model", "accuracy", "precision", "recall", "f1_score",
         "false_positive_rate", "false_negative_rate"]
    ].rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
        "false_positive_rate": "False-positive rate",
        "false_negative_rate": "False-negative rate",
    })
    return "# Test Results\n\n" + markdown_table(
        test_results, list(test_results.columns), as_percent=True
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing train.csv, validation.csv, and test.csv")
    args = parser.parse_args()

    data = load_and_label(args.data_dir)
    splits = make_model_splits(data)
    sample = choose_training_rows(splits["train"])
    validation, test = splits["validation"], splits["test"]

    # All four models run here, in the same order as their clearly labeled
    # functions above. No files are written while training or evaluating.
    results = pd.DataFrame([
        run_logistic_regression(sample, validation, test),
        run_decision_tree(sample, validation, test),
        run_random_forest(sample, validation, test),
        run_isolation_forest(sample, validation, test),
    ])

    output = Path(__file__).with_name("model_comparison.md")
    output.write_text(make_report(results), encoding="utf-8")
    print(f"Created {output}")


if __name__ == "__main__":
    main()
