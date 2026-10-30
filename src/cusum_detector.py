#!/usr/bin/env python3
"""
Streaming-style credit-card anomaly detection with the two-sided CUSUM algorithm.

The program reads transactions in timestamp order, calibrates one baseline per
customer using the first N transactions, then processes later rows one by one.
The is_anomaly column is used ONLY for evaluation after detection, never to
calculate the baseline or make a prediction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "id", "timestamp", "customer_id", "amount", "is_anomaly"
}


@dataclass
class CusumState:
    mean: float
    std: float
    s_pos: float = 0.0
    s_neg: float = 0.0
    seen: int = 0


def safe_std(values: pd.Series) -> float:
    """Return a non-zero sample standard deviation for calibration."""
    value = float(values.std(ddof=1))
    return max(value, 1e-9) if np.isfinite(value) else 1.0


def load_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount", "customer_id"]).copy()
    df = df.sort_values(["timestamp", "id"], kind="stable").reset_index(drop=True)
    return df


def calibrate(
    df: pd.DataFrame, calibration_size: int
) -> Dict[str, CusumState]:
    """Estimate each customer's baseline from their first N rows, without labels."""
    states: Dict[str, CusumState] = {}
    for customer_id, group in df.groupby("customer_id", sort=False):
        calibration = group.head(calibration_size)["amount"].astype(float)
        if len(calibration) < 2:
            raise ValueError(
                f"Customer {customer_id} has fewer than 2 calibration rows."
            )
        states[str(customer_id)] = CusumState(
            mean=float(calibration.mean()),
            std=safe_std(calibration),
        )
    return states


def run_cusum(
    df: pd.DataFrame,
    calibration_size: int = 100,
    k: float = 4.0,
    h: float = 15.0,
) -> pd.DataFrame:
    """
    Run two-sided CUSUM in timestamp order.

    Values are standardized using each customer's calibration mean and std:
        z_t = (amount_t - mean_0) / std_0
        S+ = max(0, S+_previous + z_t - k)
        S- = max(0, S-_previous - z_t - k)

    The first calibration_size rows per customer establish the baseline and
    are not scored. A later row is flagged when S+ > h or S- > h.
    """
    if calibration_size < 2:
        raise ValueError("calibration_size must be at least 2")
    if k < 0 or h <= 0:
        raise ValueError("k must be >= 0 and h must be > 0")

    states = calibrate(df, calibration_size)
    output = df.copy()
    output["cusum_z"] = np.nan
    output["cusum_positive"] = 0.0
    output["cusum_negative"] = 0.0
    output["detected_anomaly"] = 0
    output["is_calibration"] = False

    for idx, row in output.iterrows():
        state = states[str(row["customer_id"])]
        state.seen += 1
        if state.seen <= calibration_size:
            output.at[idx, "is_calibration"] = True
            continue

        z = (float(row["amount"]) - state.mean) / state.std
        state.s_pos = max(0.0, state.s_pos + z - k)
        state.s_neg = max(0.0, state.s_neg - z - k)

        output.at[idx, "cusum_z"] = z
        output.at[idx, "cusum_positive"] = state.s_pos
        output.at[idx, "cusum_negative"] = state.s_neg
        output.at[idx, "detected_anomaly"] = int(
            state.s_pos > h or state.s_neg > h
        )

    return output


def calculate_metrics(scored: pd.DataFrame) -> dict:
    """Evaluate only post-calibration rows; labels are not used by the detector."""
    test = scored.loc[~scored["is_calibration"]].copy()
    y_true = test["is_anomaly"].astype(int).to_numpy()
    y_pred = test["detected_anomaly"].astype(int).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(test) if len(test) else 0.0

    return {
        "evaluation_rows": int(len(test)),
        "actual_anomalies": int(y_true.sum()),
        "detected_anomalies": int(y_pred.sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def create_charts(scored: pd.DataFrame, metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test = scored.loc[~scored["is_calibration"]].copy()

    # 1. Transaction amount distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    normal = test.loc[test["is_anomaly"] == 0, "amount"]
    known = test.loc[test["is_anomaly"] == 1, "amount"]
    ax.hist(normal, bins=55, alpha=0.72, label="Dataset label: normal")
    ax.hist(known, bins=55, alpha=0.72, label="Dataset label: anomaly")
    ax.set_title("Transaction amount distribution")
    ax.set_xlabel("Transaction amount")
    ax.set_ylabel("Number of transactions")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "amount_distribution.png", dpi=160)
    plt.close(fig)

    # 2. CUSUM score and threshold for a customer with the most scored rows.
    chosen = test["customer_id"].value_counts().index[0]
    sample = test.loc[test["customer_id"] == chosen].sort_values("timestamp").head(350)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample["timestamp"], sample["cusum_positive"], label="CUSUM positive")
    ax.plot(sample["timestamp"], sample["cusum_negative"], label="CUSUM negative")
    ax.axhline(15.0, linestyle="--", label="Decision threshold h = 15")
    ax.set_title(f"CUSUM scores over time — customer {chosen}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative standardized deviation")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cusum_timeline.png", dpi=160)
    plt.close(fig)

    # 3. Confusion matrix
    matrix = np.array([
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)
    ax.set_title("Confusion matrix (post-calibration rows)")
    ax.set_xlabel("CUSUM prediction")
    ax.set_ylabel("Dataset reference label")
    ax.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax, label="Transactions")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit-card anomaly detection with CUSUM")
    parser.add_argument("--input", default="data/transactions_100k_clean.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--calibration-size", type=int, default=100)
    parser.add_argument("--k", type=float, default=4.0,
                        help="CUSUM tolerance in standardized units")
    parser.add_argument("--h", type=float, default=15.0,
                        help="CUSUM decision threshold")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(input_path)
    scored = run_cusum(
        transactions,
        calibration_size=args.calibration_size,
        k=args.k,
        h=args.h,
    )
    metrics = calculate_metrics(scored)

    scored.to_csv(output_dir / "scored_transactions.csv", index=False)
    anomalies = scored.loc[scored["detected_anomaly"] == 1].copy()
    anomalies.to_csv(output_dir / "detected_anomalies.csv", index=False)

    create_charts(scored, metrics, output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
