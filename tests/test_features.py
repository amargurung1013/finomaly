import numpy as np
import polars as pl

from finomaly.features import build_model_features_v2


def sample_transactions():
    return pl.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "type": [
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "CASH_IN",
            "DEBIT",
        ],
        "amount": [100, 200, 300, 400, 500],
        "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
        "oldbalanceOrg": [500, 300, 600, 800, 1000],
        "newbalanceOrig": [400, 100, 300, 1200, 500],
        "nameDest": ["M1", "C6", "C7", "C8", "M2"],
        "oldbalanceDest": [0, 50, 100, 200, 300],
        "newbalanceDest": [100, 250, 400, 600, 800],
        "isFraud": [0, 1, 0, 0, 1],
        "isFlaggedFraud": [0, 0, 0, 0, 0],
    })


def test_v2_generates_14_features():
    df = sample_transactions()

    features, feature_names = build_model_features_v2(df)

    assert features.width == 14
    assert len(feature_names) == 14


def test_labels_and_identifiers_are_excluded():
    df = sample_transactions()

    features, feature_names = build_model_features_v2(df)

    excluded_columns = {
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
    }

    assert excluded_columns.isdisjoint(feature_names)


def test_features_contain_no_invalid_values():
    df = sample_transactions()

    features, _ = build_model_features_v2(df)

    values = features.to_numpy().astype(np.float64)

    assert np.isfinite(values).all()
    assert not np.isnan(values).any()


def test_transaction_types_are_encoded_correctly():
    df = sample_transactions()

    features, _ = build_model_features_v2(df)

    assert features["type_payment"].to_list() == [1, 0, 0, 0, 0]
    assert features["type_transfer"].to_list() == [0, 1, 0, 0, 0]
    assert features["type_cash_out"].to_list() == [0, 0, 1, 0, 0]
    assert features["type_cash_in"].to_list() == [0, 0, 0, 1, 0]
    assert features["type_debit"].to_list() == [0, 0, 0, 0, 1]

