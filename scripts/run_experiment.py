import time

from finomaly.data_loader import load_transactions
from finomaly.features import (
    build_model_features,
    build_model_features_v2,
)

from finomaly.detection import (
    detect_anomalies,
    detect_anomalies_v2,
    evaluation_summary,
)

from finomaly.experiments import save_experiment_report


def run_v1(df):
    return detect_anomalies(df), build_model_features


def run_v2(df):
    return detect_anomalies_v2(df), build_model_features_v2


def run_experiment(df, version, experiment_name, algorithm, runner, notes):

    start = time.perf_counter()

    (results, model), feature_builder = runner(df)

    execution_time = time.perf_counter() - start

    metrics = evaluation_summary(results)

    _, feature_names = feature_builder(results)

    report_path = save_experiment_report(
        results=results,
        metrics=metrics,
        model=model,
        experiment_name=experiment_name,
        algorithm=algorithm,
        features=feature_names,
        execution_time=execution_time,
        notes=notes,
    )

    print(f"\n{version} completed")
    print("Metrics:", metrics)
    print("Report:", report_path)


def main():

    df = load_transactions()

    run_experiment(
        df,
        "V1",
        "V1 Isolation Forest Baseline",
        "Isolation Forest",
        run_v1,
        "Baseline using seven transaction features.",
    )

    run_experiment(
        df,
        "V2",
        "V2 Isolation Forest Improved Features",
        "Isolation Forest",
        run_v2,
        "Improved feature engineering with balance behaviour.",
    )


if __name__ == "__main__":
    main()