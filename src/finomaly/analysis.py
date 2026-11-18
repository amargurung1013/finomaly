import polars as pl


def transaction_type_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize transactions by transaction type."""

    return (
        df.group_by("type")
        .agg(
            pl.len().alias("transaction_count"),
            pl.col("amount").sum().alias("total_amount"),
            pl.col("amount").mean().alias("average_amount"),
            pl.col("isFraud").sum().alias("fraud_count"),
        )
        .sort("transaction_count", descending=True)
    )


def fraud_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize fraudulent and legitimate transactions."""

    return (
        df.group_by("isFraud")
        .agg(
            pl.len().alias("transaction_count"),
            pl.col("amount").mean().alias("average_amount"),
            pl.col("amount").median().alias("median_amount"),
        )
        .sort("isFraud")
    )


def amount_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate transaction amount distribution."""

    return df.select(
        pl.col("amount").min().alias("minimum"),
        pl.col("amount").quantile(0.25).alias("q1"),
        pl.col("amount").median().alias("median"),
        pl.col("amount").quantile(0.75).alias("q3"),
        pl.col("amount").max().alias("maximum"),
        pl.col("amount").mean().alias("mean"),
    )