import streamlit as st
import polars as pl
import plotly.graph_objects as go

from finomaly.data_loader import load_transactions, dataset_summary
from finomaly.detection import detect_anomalies_v2, evaluation_summary

from finomaly.charts import (
    transaction_type_distribution,
    fraud_distribution,
    fraud_by_transaction_type,
    fraud_rate_by_transaction_type,
    transaction_volume_over_time,
    transaction_amount_by_fraud,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Finomaly | Financial Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            padding: 18px;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.15);
        }

        h1 {
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading transaction dataset...")
def get_data():
    return load_transactions()


try:
    df = get_data()
    summary = dataset_summary(df)

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔍 Finomaly")

st.sidebar.caption(
    "Financial Intelligence & Anomaly Mining Platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transaction Analysis",
        "Fraud Analysis",
        "Anomaly Detection",
        "Model Evaluation",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Dataset Information")

st.sidebar.metric(
    "Total Transactions",
    f"{summary['total_rows']:,}",
)

st.sidebar.metric(
    "Fraud Rate",
    f"{summary['fraud_rate']:.3f}%",
)

st.sidebar.caption("Dataset: PaySim")


# ============================================================
# HEADER
# ============================================================

st.title("🔍 Finomaly")

st.caption(
    "Financial Intelligence & Anomaly Mining Platform"
)

st.divider()


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.header("Dashboard Overview")

    st.write(
        "An overview of transaction activity and fraudulent "
        "behaviour in the PaySim financial transaction dataset."
    )

    # KPI metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{summary['total_rows']:,}",
    )

    col2.metric(
        "Fraud Transactions",
        f"{summary['fraud_count']:,}",
    )

    col3.metric(
        "Legitimate Transactions",
        f"{summary['legitimate_count']:,}",
    )

    col4.metric(
        "Fraud Rate",
        f"{summary['fraud_rate']:.3f}%",
    )

    st.divider()

    # Dataset health

    st.subheader("Dataset Health")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Columns",
        summary["total_columns"],
    )

    col2.metric(
        "Missing Values",
        f"{summary['missing_values']:,}",
    )

    col3.metric(
        "Transaction Types",
        df["type"].n_unique(),
    )

    st.divider()

    # Charts

    st.subheader("Transaction Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            transaction_type_distribution(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            fraud_distribution(df),
            use_container_width=True,
        )

    st.plotly_chart(
        transaction_volume_over_time(df),
        use_container_width=True,
    )


# ============================================================
# TRANSACTION ANALYSIS PAGE
# ============================================================

elif page == "Transaction Analysis":

    st.header("Transaction Analysis")

    st.write(
        "Explore transaction patterns, transaction volumes "
        "and financial activity."
    )

    # Transaction summary

    amount_summary = df.select(
        pl.col("amount").mean().alias("average"),
        pl.col("amount").median().alias("median"),
        pl.col("amount").max().alias("maximum"),
    ).row(0, named=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Transaction",
        f"{amount_summary['average']:,.2f}",
    )

    col2.metric(
        "Median Transaction",
        f"{amount_summary['median']:,.2f}",
    )

    col3.metric(
        "Largest Transaction",
        f"{amount_summary['maximum']:,.2f}",
    )

    st.divider()

    st.subheader("Transaction Distribution")

    st.plotly_chart(
        transaction_type_distribution(df),
        use_container_width=True,
    )

    st.subheader("Transaction Volume Over Time")

    st.plotly_chart(
        transaction_volume_over_time(df),
        use_container_width=True,
    )

    st.subheader("Transaction Amount Distribution")

    st.plotly_chart(
        transaction_amount_by_fraud(df),
        use_container_width=True,
    )


# ============================================================
# FRAUD ANALYSIS PAGE
# ============================================================

elif page == "Fraud Analysis":

    st.header("Fraud Analysis")

    st.write(
        "Explore fraudulent transaction behaviour and identify "
        "which transaction categories are associated with fraud."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraudulent Transactions",
        f"{summary['fraud_count']:,}",
    )

    col2.metric(
        "Legitimate Transactions",
        f"{summary['legitimate_count']:,}",
    )

    col3.metric(
        "Fraud Rate",
        f"{summary['fraud_rate']:.4f}%",
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            fraud_distribution(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            fraud_by_transaction_type(df),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Fraud Rate by Transaction Type")

    st.plotly_chart(
        fraud_rate_by_transaction_type(df),
        use_container_width=True,
    )

    st.subheader("Transaction Amount by Fraud Status")

    st.plotly_chart(
        transaction_amount_by_fraud(df),
        use_container_width=True,
    )

    st.info(
        "Fraud count and fraud rate are different measurements. "
        "A transaction category with more fraud cases does not "
        "necessarily have a higher fraud probability."
    )


# ============================================================
# ANOMALY DETECTION PAGE
# ============================================================

elif page == "Anomaly Detection":

    st.header("Anomaly Detection")

    st.write(
        "Use Isolation Forest with V2 feature engineering "
        "to identify potentially anomalous transactions."
    )

    st.warning(
        "The model uses an unsupervised approach. It does not "
        "use the isFraud column during feature engineering or training."
    )

    st.divider()

    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:

        sample_size = st.slider(
            "Sample Size",
            min_value=10_000,
            max_value=100_000,
            value=50_000,
            step=10_000,
        )

    with col2:

        contamination = st.select_slider(
            "Contamination",
            options=[
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
            ],
            value=0.005,
        )

    st.caption(
        "Sample size controls how many transactions are analyzed. "
        "Contamination controls the expected proportion of anomalies."
    )

    if st.button(
        "Run Anomaly Detection",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("Training Isolation Forest..."):

            try:

                results, model = detect_anomalies_v2(
                    df,
                    sample_size=sample_size,
                    contamination=contamination,
                )

                st.session_state["anomaly_results"] = results

                st.session_state["model_configuration"] = {
                    "sample_size": results.height,
                    "contamination": contamination,
                }

                st.success("Anomaly detection completed!")

            except Exception as error:

                st.error(f"Model execution failed: {error}")

    # Display results if model has been executed

    if "anomaly_results" in st.session_state:

        results = st.session_state["anomaly_results"]

        metrics = evaluation_summary(results)

        st.divider()

        st.subheader("Detection Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Transactions Analyzed",
            f"{metrics['total_transactions']:,}",
        )

        col2.metric(
            "Anomalies Detected",
            f"{metrics['anomalies_detected']:,}",
        )

        col3.metric(
            "True Positives",
            f"{metrics['true_positives']:,}",
        )

        col4.metric(
            "False Positives",
            f"{metrics['false_positives']:,}",
        )

        st.divider()

        st.subheader("Highest-Ranked Anomalies")

        anomaly_results = results.filter(
            pl.col("anomaly") == 1
        )

        display_columns = [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "anomaly_score",
            "isFraud",
        ]

        st.dataframe(
            anomaly_results.select(display_columns).head(100),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = anomaly_results.write_csv()

        st.download_button(
            label="Download Detected Anomalies",
            data=csv_data,
            file_name="finomaly_detected_anomalies.csv",
            mime="text/csv",
        )

        st.caption(
            "The isFraud column is displayed only for post-detection "
            "evaluation. It is not an input to the model."
        )

    else:

        st.info(
            "Configure the model and click 'Run Anomaly Detection' "
            "to analyze transactions."
        )


# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

elif page == "Model Evaluation":

    st.header("Model Evaluation")

    st.write(
        "Evaluate Isolation Forest predictions against the "
        "known fraud labels in the dataset."
    )
    
    st.divider()
    st.subheader("V5 — Full Dataset Held-out Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", "8.82%")
    col2.metric("Recall", "33.20%")
    col3.metric("F1 Score", "13.94%")
    col4.metric("PR-AUC", "5.51%")

    st.caption(
        "Results from V5: evaluated on an independent test set "
        "of 954,393 transactions containing 1,232 fraud cases."
    )

    st.divider()

    if "anomaly_results" not in st.session_state:

        st.warning(
            "No model results available. Visit the Anomaly Detection "
            "section and run the model first."
        )

    else:

        results = st.session_state["anomaly_results"]

        metrics = evaluation_summary(results)

        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{metrics['precision']:.4f}",
        )

        col2.metric(
            "Recall",
            f"{metrics['recall']:.4f}",
        )

        col3.metric(
            "F1 Score",
            f"{metrics['f1_score']:.4f}",
        )

        st.divider()

        # Confusion matrix

        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]

        tn = (
            metrics["total_transactions"]
            - tp
            - fp
            - fn
        )

        confusion_matrix = [
            [tn, fp],
            [fn, tp],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=confusion_matrix,
                x=["Predicted Legitimate", "Predicted Fraud"],
                y=["Actual Legitimate", "Actual Fraud"],
                text=confusion_matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.divider()

        st.subheader("Detailed Evaluation")

        evaluation_data = {
            "Metric": [
                "Total Transactions",
                "Anomalies Detected",
                "Fraud Transactions",
                "True Positives",
                "False Positives",
                "False Negatives",
                "True Negatives",
                "Precision",
                "Recall",
                "F1 Score",
            ],
            "Value": [
                metrics["total_transactions"],
                metrics["anomalies_detected"],
                metrics["fraud_transactions"],
                tp,
                fp,
                fn,
                tn,
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
            ],
        }

        st.dataframe(
            pl.DataFrame(
                evaluation_data,
                schema_overrides={"Value": pl.Float64}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "These metrics are calculated on the sampled transactions "
            "used by the current detection run. They are not the "
            "held-out V4 or V5 test results."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Finomaly | Financial Intelligence & Anomaly Mining Platform"
)

st.caption(
    "Built with Python, Polars, Scikit-learn, Plotly and Streamlit."
)