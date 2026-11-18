import pytest

from finomaly.data_loader import (
    load_transactions,
    dataset_summary,
)


@pytest.fixture
def sample_csv(tmp_path):
    csv_content = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,PAYMENT,100,C1,500,400,M1,0,100,0,0
2,TRANSFER,200,C2,300,100,C3,50,250,1,0
"""

    file_path = tmp_path / "transactions.csv"
    file_path.write_text(csv_content)

    return file_path


def test_load_transactions(sample_csv):
    df = load_transactions(sample_csv)

    assert df.height == 2
    assert df.width == 11
    assert "isFraud" in df.columns


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_transactions("nonexistent.csv")


def test_missing_required_columns(tmp_path):
    csv_content = """step,type,amount
1,PAYMENT,100
"""

    file_path = tmp_path / "incomplete.csv"
    file_path.write_text(csv_content)

    with pytest.raises(ValueError):
        load_transactions(file_path)


def test_dataset_summary(sample_csv):
    df = load_transactions(sample_csv)

    summary = dataset_summary(df)

    assert summary["total_rows"] == 2
    assert summary["total_columns"] == 11
    assert summary["fraud_count"] == 1
    assert summary["legitimate_count"] == 1
    assert summary["fraud_rate"] == 50.0
    assert summary["missing_values"] == 0
