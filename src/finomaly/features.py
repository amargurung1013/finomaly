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

def build_model_features_v2(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """
    V2 feature engineering.

    Adds transaction balance behaviour and
    amount-to-balance ratios.

    Fraud labels and account identifiers are excluded.
    """

    epsilon = 1e-6

    features = df.select(
        # Transaction amount
        pl.col("amount")
        .clip(lower_bound=0)
        .log1p()
        .alias("log_amount"),

        # Transaction timing
        pl.col("step").cast(pl.Float64).alias("step"),

        (pl.col("step") % 24).alias("step_cycle"),

        # Sender balance behaviour
        (pl.col("oldbalanceOrg") - pl.col("newbalanceOrig"))
        .sign()
        .mul(
            (pl.col("oldbalanceOrg") - pl.col("newbalanceOrig"))
            .abs()
            .log1p()
        )
        .alias("origin_balance_change"),

        # Receiver balance behaviour
        (pl.col("newbalanceDest") - pl.col("oldbalanceDest"))
        .sign()
        .mul(
            (pl.col("newbalanceDest") - pl.col("oldbalanceDest"))
            .abs()
            .log1p()
        )
        .alias("destination_balance_change"),

        # Amount relative to sender's previous balance
        (
            pl.col("amount")
            / (pl.col("oldbalanceOrg") + epsilon)
        )
        .clip(0, 1000)
        .log1p()
        .alias("origin_amount_ratio"),

        # Amount relative to receiver's previous balance
        (
            pl.col("amount")
            / (pl.col("oldbalanceDest") + epsilon)
        )
        .clip(0, 1000)
        .log1p()
        .alias("destination_amount_ratio"),

        # Zero balance indicators
        (pl.col("newbalanceOrig") == 0)
        .cast(pl.Int8)
        .alias("origin_zero_balance"),

        (pl.col("newbalanceDest") == 0)
        .cast(pl.Int8)
        .alias("destination_zero_balance"),
    )

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