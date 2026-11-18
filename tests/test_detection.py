import numpy as np
import pytest
import polars as pl

from finomaly.detection import (
    detect_anomalies_v2,
    evaluation_summary,
)


def sample_transactions():
    return pl.DataFrame({
        "step": list(range(1, 101)),
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"] * 20,
        "amount": [100.0 + i * 10 for i in range(100)],
        "nameOrig": [f"C{i}" for i in range(100)],
        "oldbalanceOrg": [1000.0 + i * 20 for i in range(100)],
        "newbalanceOrig": [900.0 + i * 10 for i in range(100)],
        "nameDest": [f"M{i}" for i in range(100)],
        "oldbalanceDest": [500.0 + i * 15 for i in range(100)],
        "newbalanceDest": [600.0 + i * 20 for i in range(100)],
        "isFraud": [1 if i % 10 == 0 else 0 for i in range(100)],
        "isFlaggedFraud": [0] * 100,
    })


def test_detection_returns_requested_sample_size():
    df = sample_transactions()

    results, model = detect_anomalies_v2(
        df,
        sample_size=50,
        contamination=0.1,
    )

    assert results.height == 50


def test_predictions_and_scores_are_valid():
    df = sample_transactions()

    results, model = detect_anomalies_v2(
        df,
        sample_size=50,
        contamination=0.1,
    )

    assert set(results["anomaly"].unique().to_list()).issubset({0, 1})
    assert np.isfinite(results["anomaly_score"].to_numpy()).all()


def test_invalid_contamination_raises_error():
    df = sample_transactions()

    with pytest.raises(ValueError):
        detect_anomalies_v2(df, contamination=0)


def test_invalid_sample_size_raises_error():
    df = sample_transactions()

    with pytest.raises(ValueError):
        detect_anomalies_v2(df, sample_size=0)


def test_evaluation_summary():
    results = pl.DataFrame({
        "anomaly": [1, 1, 0, 0],
        "isFraud": [1, 0, 1, 0],
    })

    metrics = evaluation_summary(results)

    assert metrics["total_transactions"] == 4
    assert metrics["anomalies_detected"] == 2
    assert metrics["fraud_transactions"] == 2
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
