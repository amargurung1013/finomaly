import polars as pl


TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def build_model_features(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Build numerical features for anomaly detection.

    Excludes:
    - Fraud labels
    - Account identifiers
    - Account balance columns
    """

    features = df.select(
        pl.col("amount")
        .clip(lower_bound=0)
        .log1p()
        .alias("log_amount"),

        (pl.col("step") % 24).alias("step_cycle"),
    )

    # One-hot encode transaction types
    type_features = df.select(
        *[
            (pl.col("type") == transaction_type)
            .cast(pl.Int8)
            .alias(f"type_{transaction_type.lower()}")
            for transaction_type in TRANSACTION_TYPES
        ]
    )

    features = features.hstack(type_features)

    feature_names = features.columns

    return features, feature_names