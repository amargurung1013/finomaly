import time
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from finomaly.data_loader import load_transactions
from finomaly.features import build_model_features_v2
from finomaly.config import RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "logs" / "experiments"

CONTAMINATION_VALUES = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
]

N_ESTIMATORS = 150
MAX_SAMPLES = 1024


def evaluate(y_true, predictions, scores):

    return {
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, scores
        ),
        "anomalies_detected": int(predictions.sum()),
        "true_positives": int(
            ((y_true == 1) & (predictions == 1)).sum()
        ),
        "false_positives": int(
            ((y_true == 0) & (predictions == 1)).sum()
        ),
        "false_negatives": int(
            ((y_true == 1) & (predictions == 0)).sum()
        ),
    }


def main():

    print("Loading full PaySim dataset...")

    df = load_transactions()

    indices = np.arange(df.height)
    labels = df["isFraud"].to_numpy()

    # 70% training, 30% temporary
    train_idx, temporary_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    # 15% validation, 15% test
    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=0.50,
        stratify=labels[temporary_idx],
        random_state=RANDOM_STATE,
    )

    train_df = df[train_idx.tolist()]
    validation_df = df[validation_idx.tolist()]
    test_df = df[test_idx.tolist()]

    del df, indices, labels

    print("\nDataset split:")
    print("Training:", train_df.height)
    print("Validation:", validation_df.height)
    print("Testing:", test_df.height)

    print("\nBuilding V2 features...")

    X_train, feature_names = build_model_features_v2(train_df)
    X_validation, _ = build_model_features_v2(validation_df)
    X_test, _ = build_model_features_v2(test_df)

    X_train = X_train.to_numpy().astype(np.float32)
    X_validation = X_validation.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)

    y_train = train_df["isFraud"].to_numpy()
    y_validation = validation_df["isFraud"].to_numpy()
    y_test = test_df["isFraud"].to_numpy()

    if not all(
        np.isfinite(X).all()
        for X in [X_train, X_validation, X_test]
    ):
        raise ValueError("Invalid feature values detected.")

    print("\nTraining Isolation Forest...")

    start = time.perf_counter()

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train)

    training_time = time.perf_counter() - start

    print(f"Training completed in {training_time:.2f} seconds.")

    print("\nCalculating validation scores...")

    validation_scores = -model.score_samples(X_validation)

    validation_results = []

    best_f1 = -1
    best_threshold = None

    for contamination in CONTAMINATION_VALUES:

        threshold = np.quantile(
            validation_scores,
            1 - contamination,
        )

        predictions = (
            validation_scores >= threshold
        ).astype(np.int8)

        metrics = evaluate(
            y_validation,
            predictions,
            validation_scores,
        )

        validation_results.append({
            "contamination": contamination,
            "threshold": float(threshold),
            **metrics,
        })

        print(
            f"Contamination: {contamination} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = float(threshold)

    print("\nSelected validation threshold:", best_threshold)

    print("\nEvaluating on untouched test data...")

    test_start = time.perf_counter()

    test_scores = -model.score_samples(X_test)

    test_predictions = (
        test_scores >= best_threshold
    ).astype(np.int8)

    test_metrics = evaluate(
        y_test,
        test_predictions,
        test_scores,
    )

    test_time = time.perf_counter() - test_start

    print("\nTest results:")
    print(test_metrics)

    # Markdown report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_table = """
| Contamination | Precision | Recall | F1 | PR-AUC | False Positives |
|---:|---:|---:|---:|---:|---:|
"""

    for result in validation_results:
        validation_table += (
            f"| {result['contamination']} "
            f"| {result['precision']:.4f} "
            f"| {result['recall']:.4f} "
            f"| {result['f1_score']:.4f} "
            f"| {result['pr_auc']:.4f} "
            f"| {result['false_positives']} |\n"
        )

    report = f"""# V5 - Full Dataset Isolation Forest

## 1. Experiment Information

- Algorithm: Isolation Forest
- Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Dataset: Full PaySim
- Features: V2
- Random state: {RANDOM_STATE}
- Estimators: {N_ESTIMATORS}
- Max samples per tree: {MAX_SAMPLES}

## 2. Dataset Split

| Dataset | Transactions | Fraud cases |
|---|---:|---:|
| Training | {train_df.height} | {int(y_train.sum())} |
| Validation | {validation_df.height} | {int(y_validation.sum())} |
| Testing | {test_df.height} | {int(y_test.sum())} |

## 3. Features

{chr(10).join("- " + name for name in feature_names)}

## 4. Validation Results

{validation_table}

## 5. Selected Threshold

- Threshold: {best_threshold:.8f}
- Selection criterion: Highest validation F1-score

## 6. Final Test Results

| Metric | Result |
|---|---:|
| Precision | {test_metrics['precision']:.4f} |
| Recall | {test_metrics['recall']:.4f} |
| F1-score | {test_metrics['f1_score']:.4f} |
| PR-AUC | {test_metrics['pr_auc']:.4f} |
| Anomalies detected | {test_metrics['anomalies_detected']} |
| True positives | {test_metrics['true_positives']} |
| False positives | {test_metrics['false_positives']} |
| False negatives | {test_metrics['false_negatives']} |

## 7. Execution Time

- Training: {training_time:.2f} seconds
- Test scoring: {test_time:.2f} seconds

## 8. Observations

The model was trained on the training partition.
The validation partition was used to select the anomaly threshold.
The test partition was kept separate for final evaluation.

Note: This experiment uses a larger training dataset and
a different max_samples setting than V4.

---
Generated by Finomaly.
"""

    output_path = REPORT_DIR / "v5_full_dataset_isolation_forest.md"

    output_path.write_text(report, encoding="utf-8")

    print("\nReport saved:", output_path)


if __name__ == "__main__":
    main()
