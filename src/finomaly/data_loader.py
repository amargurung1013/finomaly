import polars as pl

from finomaly.config import DATASET_PATH, REQUIRED_COLUMNS


def load_transactions(path=DATASET_PATH) -> pl.DataFrame:
    """
    Load transaction data from a CSV file.
    Validate the required columns before loading.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Inspect CSV schema before loading
    schema = pl.scan_csv(path).collect_schema()

    available_columns = schema.names()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Load only the required columns
    df = pl.read_csv(
        path,
        columns=REQUIRED_COLUMNS,
        infer_schema_length=10000,
    )

    return df


def dataset_summary(df: pl.DataFrame) -> dict:
    """
    Generate basic information about the dataset.
    """

    total_rows = df.height
    total_columns = df.width

    fraud_count = df.filter(pl.col("isFraud") == 1).height

    fraud_rate = (
        (fraud_count / total_rows) * 100
        if total_rows > 0
        else 0
    )

    missing_values = (
        df.null_count()
        .sum_horizontal()
        .item()
    )

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "fraud_count": fraud_count,
        "legitimate_count": total_rows - fraud_count,
        "fraud_rate": round(fraud_rate, 4),
        "missing_values": missing_values,
    }