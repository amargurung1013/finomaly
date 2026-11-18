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
from finomaly.config import (
    DEFAULT_SAMPLE_SIZE,
    RANDOM_STATE,
    MODEL_ESTIMATORS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "logs" / "experiments"

CONTAMINATION_VALUES = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
]


def calculate_metrics(y_true, predictions, scores):

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


def format_table(rows, columns):

    header = "| " + " | ".join(columns) + " |"

    separator = "| " + " | ".join(
        ["---"] * len(columns)
    ) + " |"

    body = []

    for row in rows:
        body.append(
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
        )

    return "\n".join([header, separator, *body])


def main():

    print("Loading PaySim dataset...")

    df = load_transactions()

    # Reproducible sample
    sample = df.sample(
        n=min(DEFAULT_SAMPLE_SIZE, df.height),
        shuffle=True,
        seed=RANDOM_STATE,
    )

    labels = sample["isFraud"].to_numpy()

    # Split 70% training and 30% temporary
    indices = np.arange(sample.height)

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    # Split temporary data equally into validation and test
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        stratify=labels[temporary_indices],
        random_state=RANDOM_STATE,
    )

    train_df = sample[train_indices.tolist()]
    validation_df = sample[validation_indices.tolist()]
    test_df = sample[test_indices.tolist()]

    print("\nDataset split:")
    print("Training:", train_df.height)
    print("Validation:", validation_df.height)
    print("Testing:", test_df.height)

    # Feature engineering
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
        raise ValueError("Feature matrix contains invalid values.")

    validation_results = []

    best_f1 = -1
    best_contamination = None
    best_model = None

    print("\nStarting contamination experiments...")

    for contamination in CONTAMINATION_VALUES:

        print(f"\nTesting contamination: {contamination}")

        start = time.perf_counter()

        model = IsolationForest(
            n_estimators=MODEL_ESTIMATORS,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train)

        predictions = (model.predict(X_validation) == -1).astype(int)

        scores = -model.score_samples(X_validation)

        metrics = calculate_metrics(
            y_validation,
            predictions,
            scores,
        )

        execution_time = time.perf_counter() - start

        result = {
            "contamination": contamination,
            **metrics,
            "execution_time": round(execution_time, 2),
        }

        validation_results.append(result)

        print(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_contamination = contamination
            best_model = model

    print("\nBest validation contamination:", best_contamination)

    # Final evaluation on untouched test data
    print("\nEvaluating selected model on test data...")

    test_start = time.perf_counter()

    test_predictions = (
        best_model.predict(X_test) == -1
    ).astype(int)

    test_scores = -best_model.score_samples(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_scores,
    )

    test_execution_time = time.perf_counter() - test_start

    print("\nFinal test results:")
    print(test_metrics)

    # Report generation
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_rows = []

    for result in validation_results:

        validation_rows.append({
            "Contamination": result["contamination"],
            "Precision": round(result["precision"], 4),
            "Recall": round(result["recall"], 4),
            "F1": round(result["f1_score"], 4),
            "PR-AUC": round(result["pr_auc"], 4),
            "False Positives": result["false_positives"],
        })

    report = f"""# V4 - Isolation Forest Proper Evaluation

## 1. Experiment Information

- Algorithm: Isolation Forest
- Feature version: V2
- Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Random state: {RANDOM_STATE}
- Estimators: {MODEL_ESTIMATORS}

## 2. Dataset Split

| Dataset | Transactions | Fraud cases |
|---|---:|---:|
| Training | {train_df.height} | {int(y_train.sum())} |
| Validation | {validation_df.height} | {int(y_validation.sum())} |
| Testing | {test_df.height} | {int(y_test.sum())} |

The split is stratified by fraud labels. Labels are not used for model fitting.

## 3. Features

{chr(10).join("- " + name for name in feature_names)}

## 4. Validation Experiments

Contamination values were compared using validation F1-score.

{format_table(validation_rows, ["Contamination", "Precision", "Recall", "F1", "PR-AUC", "False Positives"])}

## 5. Selected Configuration

- Algorithm: Isolation Forest
- Contamination: {best_contamination}
- Estimators: {MODEL_ESTIMATORS}
- Selection criterion: Highest validation F1-score

## 6. Final Test Results

| Metric | Result |
|---|---:|
| Precision | {test_metrics["precision"]:.4f} |
| Recall | {test_metrics["recall"]:.4f} |
| F1-score | {test_metrics["f1_score"]:.4f} |
| PR-AUC (Average Precision) | {test_metrics["pr_auc"]:.4f} |
| Anomalies detected | {test_metrics["anomalies_detected"]} |
| True positives | {test_metrics["true_positives"]} |
| False positives | {test_metrics["false_positives"]} |
| False negatives | {test_metrics["false_negatives"]} |

## 7. Test Confusion Matrix

| | Predicted Anomaly | Predicted Normal |
|---|---:|---:|
| Actual Fraud | {test_metrics["true_positives"]} | {test_metrics["false_negatives"]} |
| Actual Legitimate | {test_metrics["false_positives"]} | {int((y_test == 0).sum()) - test_metrics["false_positives"]} |

## 8. Execution Time

Test scoring time: {test_execution_time:.2f} seconds.

## 9. Observations

The model was trained exclusively on the training partition.
Contamination was selected using validation performance.
The final metrics were calculated on an untouched test partition.

Previous V1-V3 results were exploratory in-sample experiments
and should not be directly compared with these held-out results.

---
Generated by Finomaly.
"""

    output_path = REPORT_DIR / "v4_isolation_forest_evaluation.md"

    output_path.write_text(report, encoding="utf-8")

    print("\nV4 report saved:", output_path)


if __name__ == "__main__":
    main()
