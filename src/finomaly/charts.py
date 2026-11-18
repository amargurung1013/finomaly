import polars as pl
import plotly.express as px


def transaction_type_distribution(df: pl.DataFrame):
    """Number of transactions by transaction type."""

    summary = (
        df.group_by("type")
        .len()
        .sort("len", descending=True)
    )

    fig = px.bar(
        x=summary["type"].to_list(),
        y=summary["len"].to_list(),
        labels={
            "x": "Transaction Type",
            "y": "Number of Transactions",
        },
        title="Transaction Distribution by Type",
        text_auto=True,
    )

    fig.update_layout(template="plotly_white")

    return fig


def fraud_distribution(df: pl.DataFrame):
    """Compare fraudulent and legitimate transactions."""

    summary = (
        df.group_by("isFraud")
        .len()
        .sort("isFraud")
    )

    labels = [
        "Legitimate" if value == 0 else "Fraudulent"
        for value in summary["isFraud"].to_list()
    ]

    fig = px.pie(
        names=labels,
        values=summary["len"].to_list(),
        title="Fraud vs Legitimate Transactions",
        hole=0.45,
    )

    fig.update_layout(template="plotly_white")

    return fig


def fraud_by_transaction_type(df: pl.DataFrame):
    """Number of fraudulent transactions by transaction type."""

    summary = (
        df.group_by("type")
        .agg(
            pl.col("isFraud").sum().alias("fraud_count")
        )
        .sort("fraud_count", descending=True)
    )

    fig = px.bar(
        x=summary["type"].to_list(),
        y=summary["fraud_count"].to_list(),
        labels={
            "x": "Transaction Type",
            "y": "Fraudulent Transactions",
        },
        title="Fraudulent Transactions by Type",
        text_auto=True,
    )

    fig.update_layout(template="plotly_white")

    return fig


def fraud_rate_by_transaction_type(df: pl.DataFrame):
    """Calculate fraud percentage for each transaction type."""

    summary = (
        df.group_by("type")
        .agg(
            pl.len().alias("total"),
            pl.col("isFraud").sum().alias("fraud_count"),
        )
        .with_columns(
            (
                pl.col("fraud_count")
                / pl.col("total")
                * 100
            ).alias("fraud_rate")
        )
        .sort("fraud_rate", descending=True)
    )

    fig = px.bar(
        x=summary["type"].to_list(),
        y=summary["fraud_rate"].to_list(),
        labels={
            "x": "Transaction Type",
            "y": "Fraud Rate (%)",
        },
        title="Fraud Rate by Transaction Type",
        text_auto=".3f",
    )

    fig.update_layout(template="plotly_white")

    return fig


def transaction_volume_over_time(df: pl.DataFrame):
    """Show transaction volume across simulation time steps."""

    summary = (
        df.group_by("step")
        .len()
        .sort("step")
    )

    fig = px.line(
        x=summary["step"].to_list(),
        y=summary["len"].to_list(),
        labels={
            "x": "Simulation Step",
            "y": "Transaction Volume",
        },
        title="Transaction Volume Over Time",
    )

    fig.update_layout(template="plotly_white")

    return fig


def transaction_amount_by_fraud(
    df: pl.DataFrame,
    sample_size: int = 50_000,
):
    """Compare transaction amounts for fraud and legitimate transactions."""

    sample = df.select(
        "amount",
        "isFraud",
    ).filter(
        pl.col("amount") >= 0
    )

    if sample.height > sample_size:
        sample = sample.sample(
            n=sample_size,
            shuffle=True,
            seed=42,
        )

    sample = sample.with_columns(
        pl.when(pl.col("isFraud") == 1)
        .then(pl.lit("Fraudulent"))
        .otherwise(pl.lit("Legitimate"))
        .alias("transaction_status"),

        pl.col("amount")
        .log1p()
        .alias("log_amount"),
    )

    fig = px.box(
        sample,
        x="transaction_status",
        y="log_amount",
        color="transaction_status",
        title="Transaction Amount Distribution by Fraud Status",
        labels={
            "transaction_status": "Transaction Status",
            "log_amount": "Log(1 + Transaction Amount)",
        },
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    return fig