import numpy as np
import polars as pl
from finomaly.features import build_model_features_v2
from sklearn.ensemble import IsolationForest

from finomaly.config import (
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CONTAMINATION,
    RANDOM_STATE,
    MODEL_ESTIMATORS,
)

from finomaly.features import build_model_features


def detect_anomalies(
    df: pl.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    contamination: float = DEFAULT_CONTAMINATION,
) -> tuple[pl.DataFrame, IsolationForest]:

    """
    Detect anomalous transactions using Isolation Forest.

    Fraud labels are excluded from model training.
    """

    if not 0 < contamination <= 0.5:
        raise ValueError("Contamination must be between 0 and 0.5.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

    # Reproducible sampling
    sample_size = min(sample_size, df.height)

    sample = df.sample(
        n=sample_size,
        shuffle=True,
        seed=RANDOM_STATE,
    )

    # Feature engineering
    X, feature_names = build_model_features(sample)

    X_array = X.to_numpy().astype(np.float32)

    if not np.isfinite(X_array).all():
        raise ValueError("Feature matrix contains invalid values.")

    # Initialize Isolation Forest
    model = IsolationForest(
        n_estimators=MODEL_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Train without fraud labels
    model.fit(X_array)

    # Predictions: -1 = anomaly, 1 = normal
    predictions = model.predict(X_array)

    # Higher score = more anomalous
    anomaly_scores = -model.score_samples(X_array)

    # Attach predictions to sampled transactions
    results = sample.with_columns(
        pl.Series(
            "anomaly",
            (predictions == -1).astype(np.int8),
        ),
        pl.Series(
            "anomaly_score",
            anomaly_scores,
        ),
    )

    results = results.sort(
        "anomaly_score",
        descending=True,
    )

    return results, model

def detect_anomalies_v2(
    df: pl.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    contamination: float = DEFAULT_CONTAMINATION,
) -> tuple[pl.DataFrame, IsolationForest]:

    if not 0 < contamination <= 0.5:
        raise ValueError("Contamination must be between 0 and 0.5.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

    sample_size = min(sample_size, df.height)

    sample = df.sample(
        n=sample_size,
        shuffle=True,
        seed=RANDOM_STATE,
    )

    X, feature_names = build_model_features_v2(sample)

    X_array = X.to_numpy().astype(np.float32)

    if not np.isfinite(X_array).all():
        raise ValueError("Feature matrix contains invalid values.")

    model = IsolationForest(
        n_estimators=MODEL_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_array)

    predictions = model.predict(X_array)
    anomaly_scores = -model.score_samples(X_array)

    results = sample.with_columns(
        pl.Series("anomaly", (predictions == -1).astype(np.int8)),
        pl.Series("anomaly_score", anomaly_scores),
    )

    results = results.sort(
        "anomaly_score",
        descending=True,
    )

    return results, model

def evaluation_summary(
    results: pl.DataFrame,
) -> dict:

    """
    Evaluate anomaly predictions against known fraud labels.

    The fraud labels are used only after model training.
    """

    true_positive = results.filter(
        (pl.col("anomaly") == 1)
        & (pl.col("isFraud") == 1)
    ).height

    false_positive = results.filter(
        (pl.col("anomaly") == 1)
        & (pl.col("isFraud") == 0)
    ).height

    false_negative = results.filter(
        (pl.col("anomaly") == 0)
        & (pl.col("isFraud") == 1)
    ).height

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "total_transactions": results.height,
        "anomalies_detected": true_positive + false_positive,
        "fraud_transactions": true_positive + false_negative,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
    }