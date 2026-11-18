import polars as pl

from finomaly.charts import (
    transaction_type_distribution,
    fraud_distribution,
    fraud_by_transaction_type,
    fraud_rate_by_transaction_type,
    transaction_volume_over_time,
    transaction_amount_by_fraud,
)


def sample_transactions():
    return pl.DataFrame({
        "step": [1, 1, 2, 2, 3, 3],
        "type": [
            "PAYMENT",
            "TRANSFER",
            "PAYMENT",
            "CASH_OUT",
            "TRANSFER",
            "CASH_OUT",
        ],
        "amount": [100, 200, 150, 300, 500, 250],
        "isFraud": [0, 1, 0, 1, 1, 0],
    })


def test_transaction_type_distribution():
    fig = transaction_type_distribution(sample_transactions())

    assert fig is not None
    assert fig.layout.title.text == "Transaction Distribution by Type"
    assert len(fig.data[0].x) == 3


def test_fraud_distribution():
    fig = fraud_distribution(sample_transactions())

    assert fig is not None
    assert fig.layout.title.text == "Fraud vs Legitimate Transactions"
    assert len(fig.data[0].labels) == 2


def test_fraud_by_transaction_type():
    fig = fraud_by_transaction_type(sample_transactions())

    assert fig is not None
    assert fig.layout.title.text == "Fraudulent Transactions by Type"


def test_fraud_rate_by_transaction_type():
    fig = fraud_rate_by_transaction_type(sample_transactions())

    assert fig is not None
    assert fig.layout.title.text == "Fraud Rate by Transaction Type"


def test_transaction_volume_over_time():
    fig = transaction_volume_over_time(sample_transactions())

    assert fig is not None
    assert fig.layout.title.text == "Transaction Volume Over Time"
    assert len(fig.data[0].x) == 3


def test_transaction_amount_by_fraud():
    fig = transaction_amount_by_fraud(sample_transactions())

    assert fig is not None
    assert fig.layout.title.text == (
        "Transaction Amount Distribution by Fraud Status"
    )