import time

import polars as pl

from finomaly.data_loader import load_transactions
from finomaly.features import build_model_features_v2
from finomaly.detection import evaluation_summary
from finomaly.config import (
    DEFAULT_SAMPLE_SIZE,
    RANDOM_STATE,
    MODEL_ESTIMATORS,
)

from sklearn.ensemble import IsolationForest


CONTAMINATION_VALUES = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
]


def main():

    df = load_transactions()

    sample = df.sample(
        n=min(DEFAULT_SAMPLE_SIZE, df.height),
        shuffle=True,
        seed=RANDOM_STATE,
    )

    X, feature_names = build_model_features_v2(sample)

    X_array = X.to_numpy()

    experiment_results = []

    for contamination in CONTAMINATION_VALUES:

        print(f"\nTesting contamination: {contamination}")

        start = time.perf_counter()

        model = IsolationForest(
            n_estimators=MODEL_ESTIMATORS,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_array)

        predictions = model.predict(X_array)

        scores = -model.score_samples(X_array)

        results = sample.with_columns(
            pl.Series(
                "anomaly",
                (predictions == -1).astype("int8"),
            ),
            pl.Series("anomaly_score", scores),
        )

        execution_time = time.perf_counter() - start

        metrics = evaluation_summary(results)

        experiment_results.append({
            "contamination": contamination,
            **metrics,
            "execution_time": round(execution_time, 2),
        })

        print(metrics)

    comparison = pl.DataFrame(experiment_results)

    output_path = (
        "logs/experiments/v3_contamination_comparison.md"
    )

    markdown = comparison.to_pandas().to_markdown(index=False)

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("# V3 - Contamination Comparison\n\n")

        file.write(
            "Isolation Forest with V2 features. "
            "Only contamination changes between experiments.\n\n"
        )

        file.write(markdown)

    print("\nComparison saved:", output_path)


if __name__ == "__main__":
    main()
