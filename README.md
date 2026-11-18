# 🔍 Finomaly
### Financial Intelligence & Anomaly Mining Platform

**An exploratory financial transaction analysis and anomaly-detection platform built using Python, Polars, Scikit-learn, Plotly, and Streamlit.**

Finomaly is a data-driven financial intelligence platform designed to explore transaction behaviour, identify unusual financial activity, and investigate potentially fraudulent transactions using unsupervised machine learning.

The project uses the PaySim financial transaction dataset and applies the Isolation Forest algorithm to identify transactions that exhibit unusual behavioural patterns.

It combines data processing, exploratory data analysis, feature engineering, machine learning experimentation, model evaluation, and interactive visualisation in a single application.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Project Objectives](#3-project-objectives)
- [4. Key Features](#4-key-features)
- [5. Technology Stack](#5-technology-stack)
- [6. Dataset](#6-dataset)
- [7. System Architecture](#7-system-architecture)
- [8. Data Processing](#8-data-processing)
- [9. Exploratory Data Analysis](#9-exploratory-data-analysis)
- [10. Feature Engineering](#10-feature-engineering)
- [11. Anomaly Detection Methodology](#11-anomaly-detection-methodology)
- [12. Experimentation and Model Development](#12-experimentation-and-model-development)
- [13. Model Evaluation](#13-model-evaluation)
- [14. Interactive Dashboard](#14-interactive-dashboard)
- [15. Project Structure](#15-project-structure)
- [16. Installation and Setup](#16-installation-and-setup)
- [17. Running the Application](#17-running-the-application)
- [18. Running Experiments](#18-running-experiments)
- [19. Running Tests](#19-running-tests)
- [20. Key Findings](#20-key-findings)
- [21. Limitations](#21-limitations)
- [22. Future Improvements](#22-future-improvements)
- [23. Contributors](#23-contributors)

---

# 1. Project Overview

Financial institutions process enormous volumes of transactions every day. Identifying unusual transactions within these datasets is challenging, particularly when fraudulent transactions represent only a very small proportion of total activity.

Traditional rule-based systems depend on predefined conditions. Although useful, these systems may struggle to identify unusual transaction patterns that have not already been anticipated.

Finomaly explores an alternative approach through **unsupervised anomaly detection**.

The platform uses Isolation Forest, an unsupervised machine learning algorithm, to identify transactions that differ significantly from the general distribution of transaction behaviour.

The application provides:

- Financial transaction data exploration.
- Fraud distribution analysis.
- Transaction behaviour visualisation.
- Feature engineering for machine learning.
- Unsupervised anomaly detection.
- Model performance evaluation.
- Interactive anomaly inspection.
- Experiment tracking and reproducibility.

The goal is to demonstrate how data engineering, statistical analysis, and machine learning can be combined to investigate potentially suspicious financial activity.

# 2. Problem Statement

Financial transaction datasets frequently contain highly imbalanced distributions.

In a typical financial dataset, legitimate transactions significantly outnumber fraudulent transactions. This creates several challenges:

1. Fraudulent transactions are relatively rare.
2. Unusual transactions may not always be fraudulent.
3. Legitimate transactions can exhibit unusual characteristics.
4. Detecting suspicious behaviour without relying on predefined fraud labels is difficult.
5. High false-positive rates can make anomaly detection systems difficult to use operationally.

Finomaly investigates whether an unsupervised machine learning algorithm can identify unusual transaction patterns without using known fraud labels during model training.

The project subsequently compares detected anomalies against available fraud labels to evaluate the effectiveness of the approach.

# 3. Project Objectives

The primary objectives are:

- Load and process a large financial transaction dataset efficiently.
- Examine transaction distributions and financial behaviour.
- Investigate relationships between transaction characteristics and fraudulent activity.
- Develop meaningful numerical features for anomaly detection.
- Implement an Isolation Forest-based anomaly detection pipeline.
- Experiment with feature engineering and anomaly thresholds.
- Evaluate detection performance using precision, recall, F1-score, and Precision-Recall AUC.
- Build an interactive dashboard for exploring transactions and model predictions.
- Maintain reproducible experiments and automated tests.

# 4. Key Features

| Feature | Description |
|---|---|
| Transaction Analysis | Examines transaction volume, amounts, and transaction categories. |
| Fraud Analysis | Investigates fraud distribution and fraud rates across transaction types. |
| Feature Engineering | Transforms raw transaction attributes into machine learning features. |
| Anomaly Detection | Uses Isolation Forest to identify unusual transactions. |
| Model Evaluation | Calculates classification metrics against known fraud labels. |
| Confusion Matrix | Visualises true positives, false positives, true negatives, and false negatives. |
| Interactive Dashboard | Provides an interface for exploring data and model results. |
| CSV Export | Allows users to download detected anomalies. |
| Experiment Tracking | Stores experiment configurations, results, and observations. |
| Automated Testing | Validates important data processing, feature engineering, detection, and chart functionality. |

# 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Polars | Efficient DataFrame operations and large-scale data processing |
| NumPy | Numerical operations |
| Pandas | Data analysis dependency |
| PyArrow | Columnar data interoperability |
| Scikit-learn | Isolation Forest and model evaluation |
| Plotly | Interactive charts and visualisations |
| Streamlit | Interactive dashboard |
| Pytest | Automated testing |
| uv | Python dependency and environment management |
| Git | Version control |
| GitHub | Source code hosting |

### Why Polars?

The PaySim dataset contains more than six million transaction records.

Polars was selected for its efficient columnar execution model, expressive DataFrame API, and suitability for processing larger datasets.

It is used for loading, filtering, aggregating, transforming, and analysing transaction data.

### Why Isolation Forest?

Isolation Forest is an unsupervised anomaly-detection algorithm.

Unlike conventional supervised classification approaches, it does not require fraud labels to train the model.

It works by randomly selecting features and split values to isolate observations. Anomalous observations tend to be isolated using fewer splits than ordinary observations.

This makes it suitable for exploring unusual patterns in financial transaction data.

# 6. Dataset

Finomaly uses the **PaySim synthetic financial transaction dataset**.

The dataset simulates mobile money transactions and contains transaction-level information, account balance changes, transaction categories, and fraud labels.

### Dataset statistics

| Property | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Total columns | 11 |
| Fraudulent transactions | 8,213 |
| Legitimate transactions | 6,354,407 |
| Fraud rate | Approximately 0.1291% |
| Missing values | 0 |
| Transaction categories | 5 |

### Dataset columns

| Column | Description |
|---|---|
| `step` | Time step associated with the transaction |
| `type` | Transaction category |
| `amount` | Transaction amount |
| `nameOrig` | Originating account identifier |
| `oldbalanceOrg` | Originating account balance before the transaction |
| `newbalanceOrig` | Originating account balance after the transaction |
| `nameDest` | Destination account identifier |
| `oldbalanceDest` | Destination account balance before the transaction |
| `newbalanceDest` | Destination account balance after the transaction |
| `isFraud` | Indicates whether the transaction is fraudulent |
| `isFlaggedFraud` | Indicates whether the transaction was flagged by the dataset's original rule-based system |

### Transaction categories

The dataset contains five transaction types:

- `CASH_IN`
- `CASH_OUT`
- `DEBIT`
- `PAYMENT`
- `TRANSFER`

### Dataset imbalance

Only approximately 0.1291% of the transactions are fraudulent.

This means that a model that simply predicts every transaction as legitimate would achieve extremely high accuracy while detecting no fraud.

Consequently, accuracy alone is not an appropriate metric for evaluating this project.

Finomaly instead focuses on precision, recall, F1-score, and Precision-Recall AUC.

### Obtaining the dataset

The dataset is not included in the Git repository.

Download the PaySim dataset separately and place the extracted CSV file inside:

```text
data/raw/
```

The expected filename is:

```text
PS_20174392719_1491204439457_log.csv
```

The application expects the following path:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

# 7. System Architecture

Finomaly follows a modular architecture in which data processing, feature engineering, anomaly detection, analysis, visualisation, and application logic are separated into different modules.

### High-level workflow

```text
                 PaySim Dataset
                        |
                        v
                 Data Loading
                        |
                        v
               Data Validation
                        |
                        v
              Transaction Analysis
                        |
                        v
               Feature Engineering
                        |
                        v
                Isolation Forest
                        |
                        v
               Anomaly Detection
                        |
                        v
                Model Evaluation
                        |
                        v
               Interactive Dashboard
                        |
                        v
              Results and CSV Export
```

### Architectural components

**1. Data Layer**

Responsible for loading the dataset, validating its structure, and generating basic dataset statistics.

**2. Analysis Layer**

Performs transaction-level aggregations and investigates fraud distribution, transaction types, and transaction amounts.

**3. Feature Engineering Layer**

Transforms raw transaction attributes into numerical features suitable for machine learning.

**4. Detection Layer**

Implements the Isolation Forest algorithm and generates anomaly predictions and anomaly scores.

**5. Experiment Layer**

Records experiment configurations, evaluation results, and observations.

**6. Visualisation Layer**

Generates interactive charts using Plotly.

**7. Application Layer**

Uses Streamlit to bring the different components together into an interactive dashboard.

# 8. Data Processing

The data processing pipeline is implemented using Polars.

The dataset loader performs several operations:

1. Checks whether the dataset exists.
2. Validates the presence of required columns.
3. Loads the required transaction columns.
4. Returns a Polars DataFrame.
5. Generates dataset-level statistics.

The application also calculates:

- Total number of transactions.
- Number of fraudulent transactions.
- Number of legitimate transactions.
- Fraud percentage.
- Number of columns.
- Missing-value count.

The loader is separated from the application so that data validation and loading can be tested independently.

# 9. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to understand the structure and behaviour of the dataset before applying machine learning.

Finomaly investigates transaction activity from several perspectives.

### 9.1 Transaction distribution

Transactions are grouped by transaction type to understand the distribution of financial activity.

The analysis examines:

- Transaction counts.
- Total transaction amounts.
- Average transaction amounts.
- Fraud counts by transaction category.

### 9.2 Fraud distribution

The dataset is examined to determine the proportion of fraudulent and legitimate transactions.

This reveals the severe class imbalance present in the dataset.

### 9.3 Fraud by transaction type

The project investigates which transaction categories contain fraudulent transactions.

The dataset contains fraudulent activity in:

- `CASH_OUT`
- `TRANSFER`

The observed fraud counts are:

| Transaction Type | Fraudulent Transactions |
|---|---:|
| CASH_OUT | 4,116 |
| TRANSFER | 4,097 |
| CASH_IN | 0 |
| PAYMENT | 0 |
| DEBIT | 0 |

### 9.4 Fraud rate by transaction type

Fraud count and fraud rate are analysed separately.

A transaction category with a higher total fraud count does not necessarily have a higher probability of fraud.

The fraud rate provides additional context by considering the number of transactions in each category.

### 9.5 Transaction amount analysis

Transaction amounts are investigated using descriptive statistics.

The analysis includes:

- Minimum transaction amount.
- First quartile.
- Median.
- Third quartile.
- Maximum transaction amount.
- Mean transaction amount.

Fraudulent and legitimate transaction amounts are also compared using visualisations.

### 9.6 Transaction volume over time

The `step` column is used to examine transaction activity across simulation time.

This helps identify changes in transaction volume and potential temporal patterns.

# 10. Feature Engineering

Feature engineering transforms raw transaction information into numerical representations that can be processed by the Isolation Forest algorithm.

Finomaly explores two feature sets.

## V1 — Baseline features

The baseline feature set contains seven features:

1. `log_amount`
2. `step_cycle`
3. `type_CASH_IN`
4. `type_CASH_OUT`
5. `type_DEBIT`
6. `type_PAYMENT`
7. `type_TRANSFER`

The transaction amount is log-transformed to reduce the influence of extreme values.

Transaction types are one-hot encoded.

## V2 — Improved feature set

The second feature set introduces additional balance-related and transaction-behaviour features.

It contains 14 features:

| Feature | Description |
|---|---|
| `log_amount` | Log-transformed transaction amount |
| `step` | Original transaction time step |
| `step_cycle` | Cyclical representation of transaction time |
| `origin_balance_change` | Signed log-transformed difference between originating balance before and after the transaction |
| `destination_balance_change` | Signed log-transformed difference between destination balance after and before the transaction |
| `origin_amount_ratio` | Transaction amount relative to the originating account's previous balance |
| `destination_amount_ratio` | Transaction amount relative to the destination account's previous balance |
| `origin_zero_balance` | Indicates whether the originating account had a zero balance before the transaction |
| `destination_zero_balance` | Indicates whether the destination account had a zero balance before the transaction |
| `type_CASH_IN` | One-hot encoded CASH_IN transaction indicator |
| `type_CASH_OUT` | One-hot encoded CASH_OUT transaction indicator |
| `type_DEBIT` | One-hot encoded DEBIT transaction indicator |
| `type_PAYMENT` | One-hot encoded PAYMENT transaction indicator |
| `type_TRANSFER` | One-hot encoded TRANSFER transaction indicator |

### Why introduce balance-related features?

Transaction amounts alone may not provide sufficient information about financial behaviour.

For example, a transaction of a particular amount may be ordinary for one account but unusual for another.

Balance changes and transaction-to-balance ratios provide additional context about the financial effect of each transaction.

The V2 feature set therefore explores whether account balance behaviour improves anomaly detection.

### Important design decision

The following columns are excluded from model features:

- `isFraud`
- `isFlaggedFraud`
- `nameOrig`
- `nameDest`

Fraud labels are excluded because the model is intended to learn transaction patterns without supervised fraud labels.

Account identifiers are excluded because they are identifiers rather than direct measures of transaction behaviour.

The fraud label is retained separately for evaluation after anomaly detection.

# 11. Anomaly Detection Methodology

Finomaly uses the Isolation Forest algorithm from Scikit-learn.

Isolation Forest identifies anomalies by isolating observations through random decision-tree splits.

### How Isolation Forest works

The algorithm operates using the following general process:

1. Randomly selects a feature.
2. Randomly selects a split value within the feature's range.
3. Divides the observations into smaller groups.
4. Repeats the splitting process until observations are isolated or a stopping condition is reached.
5. Examines how many splits were required to isolate each observation.

Observations that are easier to isolate are considered more anomalous.

The algorithm produces anomaly scores that can be used to rank transactions according to their unusualness.

### Model configuration

The project uses the following main configuration:

| Parameter | Value |
|---|---:|
| Algorithm | Isolation Forest |
| Number of estimators | 150 |
| Maximum samples per tree | 1,024 in V5 |
| Random state | 42 |
| Parallel processing | Enabled |
| Feature set | V2 |
| Contamination | Selected through validation threshold experiments in V5 |

### Contamination

Contamination controls the expected proportion of observations classified as anomalies when using the conventional Isolation Forest contamination threshold.

Finomaly investigates several contamination values:

```text
0.001
0.002
0.005
0.010
0.020
0.050
```

A lower threshold generally identifies fewer transactions as anomalous.

A higher threshold identifies more transactions but can also increase false positives.

The project investigates this trade-off through experimentation.

### Anomaly scores

The application produces an anomaly score for each analysed transaction.

Higher anomaly scores indicate observations considered more unusual by the model.

The score is used to rank detected anomalies and display the highest-ranked transactions.

# 12. Experimentation and Model Development

Finomaly follows an iterative experimentation process.

Each experiment investigates a particular modelling decision and records its results.

The experiments are stored in:

```text
logs/experiments/
```

## V1 — Isolation Forest baseline

The first experiment establishes a baseline using the original seven-feature representation.

Configuration:

- Sample size: 100,000 transactions.
- Features: V1.
- Estimators: 150.
- Contamination: 0.01.
- Random state: 42.

Results:

| Metric | Result |
|---|---:|
| Precision | 0.0231 |
| Recall | 0.1797 |
| F1-score | 0.0409 |

The baseline model identifies some fraudulent transactions but generates a large number of false positives.

## V2 — Improved feature engineering

The second experiment introduces the additional balance-related features.

Configuration:

- Sample size: 100,000 transactions.
- Features: V2.
- Estimators: 150.
- Contamination: 0.01.
- Random state: 42.

Results:

| Metric | V1 | V2 |
|---|---:|---:|
| Precision | 0.0231 | 0.0540 |
| Recall | 0.1797 | 0.4219 |
| F1-score | 0.0409 | 0.0957 |

The additional features improve the observed precision, recall, and F1-score in this experiment.

However, these are exploratory in-sample results and should not be interpreted as independent test performance.

## V3 — Contamination comparison

The third experiment investigates how contamination affects anomaly detection.

The V2 feature set is retained while the contamination parameter is changed.

| Contamination | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.001 | 0.2400 | 0.1875 | 0.2105 |
| 0.002 | 0.1750 | 0.2734 | 0.2134 |
| 0.005 | 0.1000 | 0.3906 | 0.1592 |
| 0.010 | 0.0540 | 0.4219 | 0.0957 |
| 0.020 | 0.0315 | 0.4922 | 0.0592 |
| 0.050 | 0.0138 | 0.5391 | 0.0269 |

The experiment demonstrates the trade-off between precision and recall.

As contamination increases, the model detects more fraudulent transactions but also flags more legitimate transactions.

The best observed F1-score in this experiment is approximately 0.2134 at contamination 0.002.

These results are in-sample and are not used as the final generalisation estimate.

## V4 — Held-out evaluation on a sample

The fourth experiment introduces a training, validation, and testing workflow.

A reproducible sample of 100,000 transactions is divided into:

- 70% training.
- 15% validation.
- 15% testing.

The validation set is used to select a threshold, and the test set is used for final evaluation.

The resulting test performance was:

| Metric | Result |
|---|---:|
| Precision | 0.1429 |
| Recall | 0.1053 |
| F1-score | 0.1212 |
| PR-AUC | 0.0912 |

The small number of fraud cases in the test split makes these results sensitive to individual predictions.

This motivated the next experiment.

## V5 — Full dataset held-out evaluation

The fifth experiment extends the evaluation to the complete dataset.

The dataset is divided using a stratified 70/15/15 split.

| Split | Transactions | Fraud Cases |
|---|---:|---:|
| Training | 4,453,834 | 5,749 |
| Validation | 954,393 | 1,232 |
| Testing | 954,393 | 1,232 |

The model is trained on the training split.

Validation data is used to select an anomaly-score threshold.

The selected threshold is then applied to the independent test split.

### V5 configuration

| Parameter | Value |
|---|---|
| Algorithm | Isolation Forest |
| Features | V2 |
| Estimators | 150 |
| Maximum samples | 1,024 |
| Contamination during fitting | `auto` |
| Random state | 42 |
| Threshold selection | Validation F1-score |

The V5 experiment uses anomaly scores and validation-based threshold selection rather than repeatedly fitting separate models for each threshold.

The best validation F1-score was observed at the 0.005 threshold configuration.

### V5 final test results

| Metric | Result |
|---|---:|
| Precision | 0.0882 |
| Recall | 0.3320 |
| F1-score | 0.1394 |
| PR-AUC | 0.0551 |
| True Positives | 409 |
| False Positives | 4,229 |
| False Negatives | 823 |
| True Negatives | 948,932 |
| Anomalies Detected | 4,638 |

The model identifies 409 fraudulent transactions in the held-out test set.

However, it also flags 4,229 legitimate transactions as anomalous.

This highlights an important limitation of the current approach: identifying unusual financial behaviour does not necessarily mean identifying fraud.

V5 is the primary held-out evaluation result used by the project.

# 13. Model Evaluation

Finomaly evaluates anomaly detection using four primary metrics.

## Precision

Precision measures the proportion of detected anomalies that are actually fraudulent.

\[
Precision = \frac{TP}{TP + FP}
\]

A higher precision indicates fewer false-positive predictions among flagged transactions.

## Recall

Recall measures the proportion of actual fraudulent transactions successfully detected.

\[
Recall = \frac{TP}{TP + FN}
\]

A higher recall indicates that the model detects a larger proportion of fraudulent activity.

## F1-score

F1-score is the harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

It provides a combined measure of the trade-off between precision and recall.

## Precision-Recall AUC

Precision-Recall AUC measures the relationship between precision and recall across different decision thresholds.

It is particularly useful for imbalanced classification problems because it focuses on the positive class.

### Why not accuracy?

The PaySim dataset contains approximately 0.1291% fraudulent transactions.

A model predicting every transaction as legitimate would still achieve very high accuracy.

However, it would fail to detect fraud entirely.

For this reason, Finomaly prioritises precision, recall, F1-score, and PR-AUC over accuracy.

### Understanding the confusion matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | True Negative | False Positive |
| Actual Fraud | False Negative | True Positive |

These four values help explain the types of mistakes made by the model.

# 14. Interactive Dashboard

Finomaly includes an interactive web application built using Streamlit.

Start the application to explore the different sections.

## 14.1 Overview

The Overview page provides a high-level summary of the PaySim dataset.

It displays:

- Total transactions.
- Fraudulent transactions.
- Legitimate transactions.
- Fraud percentage.
- Dataset column count.
- Missing values.
- Number of transaction categories.

It also includes visualisations of transaction distribution, fraud distribution, and transaction volume over time.

## 14.2 Transaction Analysis

The Transaction Analysis page explores general financial activity.

It displays:

- Average transaction amount.
- Median transaction amount.
- Maximum transaction amount.
- Transaction distribution by category.
- Transaction volume over time.
- Transaction amount distribution by fraud status.

This section helps users understand the overall characteristics of the dataset.

## 14.3 Fraud Analysis

The Fraud Analysis page focuses on fraudulent transaction behaviour.

It includes:

- Fraudulent transaction count.
- Legitimate transaction count.
- Fraud percentage.
- Fraud distribution.
- Fraud count by transaction type.
- Fraud rate by transaction type.
- Transaction amount comparison by fraud status.

The distinction between fraud count and fraud rate is important when interpreting transaction categories.

## 14.4 Anomaly Detection

The Anomaly Detection page allows users to run Isolation Forest interactively.

Users can configure:

- Sample size.
- Contamination.

The application then:

1. Builds the V2 feature representation.
2. Runs Isolation Forest.
3. Generates anomaly predictions.
4. Calculates anomaly scores.
5. Compares predictions against known fraud labels for evaluation.
6. Displays detection statistics.
7. Ranks the highest-scoring anomalous transactions.

The application displays selected transaction information, including:

- Transaction step.
- Transaction type.
- Transaction amount.
- Originating account balances.
- Destination account balances.
- Anomaly score.
- Fraud label for post-detection evaluation.

The fraud label is not used as an input to the model.

Users can also download the detected anomalies as a CSV file.

## 14.5 Model Evaluation

The Model Evaluation page provides two different perspectives.

**V5 Held-out Evaluation**

Displays the full-dataset V5 test metrics:

- Precision.
- Recall.
- F1-score.
- PR-AUC.

These results represent performance on the independent V5 test split.

**Interactive Evaluation**

Displays metrics from the current dashboard detection run.

It includes:

- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Detailed evaluation statistics.

These results are based on the sample used in the interactive detection run.

They should not be confused with the independent V5 test results.

# 15. Project Structure

```text
finomaly/
│
├── data/
│   ├── raw/
│   │   ├── paysim.zip
│   │   └── PS_20174392719_1491204439457_log.csv
│   │
│   └── processed/
│
├── logs/
│   └── experiments/
│       ├── v1_isolation_forest_baseline.md
│       ├── v2_isolation_forest_improved_features.md
│       ├── v3_contamination_comparison.md
│       ├── v4_isolation_forest_evaluation.md
│       └── v5_full_dataset_isolation_forest.md
│
├── notebooks/
│
├── scripts/
│   ├── v1_v2_baseline_experiments.py
│   ├── v3_contamination_comparison.py
│   ├── v4_sampled_evaluation.py
│   └── v5_full_dataset_evaluation.py
│
├── src/
│   └── finomaly/
│       ├── __init__.py
│       ├── analysis.py
│       ├── app.py
│       ├── charts.py
│       ├── config.py
│       ├── data_loader.py
│       ├── detection.py
│       ├── experiments.py
│       └── features.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_detection.py
│   ├── test_features.py
│   └── test_charts.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Stores project paths, dataset configuration, and model defaults. |
| `data_loader.py` | Loads and validates the PaySim dataset. |
| `analysis.py` | Performs transaction and fraud analysis. |
| `features.py` | Builds V1 and V2 machine learning features. |
| `detection.py` | Implements Isolation Forest anomaly detection and evaluation metrics. |
| `experiments.py` | Generates experiment reports. |
| `charts.py` | Creates interactive Plotly visualisations. |
| `app.py` | Implements the Streamlit dashboard. |

The `scripts/` directory contains standalone experiment execution scripts.

The `tests/` directory contains automated tests for important project components.

# 16. Installation and Setup

## Prerequisites

Make sure the following are installed:

- Python 3.12 or compatible Python version.
- uv.
- Git.

### Step 1 — Clone the repository

```bash
git clone https://github.com/amargurung1013/finomaly.git
```

Navigate into the project:

```bash
cd finomaly
```

### Step 2 — Install dependencies

```bash
uv sync --dev
```

This creates or updates the project's virtual environment and installs the dependencies declared in `pyproject.toml`.

### Step 3 — Add the dataset

Place the extracted PaySim CSV file inside:

```text
data/raw/
```

Ensure that the filename is:

```text
PS_20174392719_1491204439457_log.csv
```

The application will not run correctly without the dataset.

# 17. Running the Application

Start the Streamlit dashboard:

```bash
uv run streamlit run src/finomaly/app.py
```

Streamlit will start a local development server.

Open the local address displayed in your terminal, usually:

```text
http://localhost:8501
```

Use the sidebar to navigate between the five dashboard sections.

# 18. Running Experiments

The project includes standalone scripts for executing model experiments.

Run scripts from the project root.

### Baseline experiment

```bash
uv run python scripts/v1_v2_baseline_experiments.py
│   ├── v3_contamination_comparison.py
```

### Contamination comparison

```bash
uv run python scripts/v3_contamination_comparison.py
```

### V4 held-out evaluation

```bash
uv run python scripts/v4_sampled_evaluation.py
```

### V5 full-dataset evaluation

```bash
uv run python scripts/v5_full_dataset_evaluation.py
```

Experiment reports are stored in:

```text
logs/experiments/
```

These reports preserve configurations, metrics, and observations for reproducibility.

# 19. Running Tests

Finomaly uses Pytest for automated testing.

Run the complete test suite:

```bash
uv run pytest -q
```

The test suite covers:

- Dataset loading and validation.
- Feature engineering.
- Anomaly detection.
- Evaluation functionality.
- Chart generation.

The project currently contains 19 automated tests.

Expected output:

```text
19 passed
```

The tests use synthetic data and temporary files where appropriate, allowing important functionality to be validated without repeatedly processing the entire PaySim dataset.

# 20. Key Findings

Several observations emerged during the development of Finomaly.

### 1. Severe class imbalance

Fraudulent transactions represent approximately 0.1291% of the dataset.

This makes fraud detection challenging and makes accuracy an insufficient evaluation metric.

### 2. Transaction categories matter

The dataset contains fraudulent transactions in CASH_OUT and TRANSFER operations.

Understanding transaction categories provides useful context for investigating suspicious activity.

### 3. Feature engineering improves exploratory results

The V2 feature representation produced better observed results than the V1 baseline in the initial experiments.

This suggests that balance-related features may provide useful information about transaction behaviour.

### 4. Contamination creates a precision-recall trade-off

Increasing the number of observations classified as anomalous can improve recall but may significantly reduce precision.

Selecting a threshold is therefore an important part of anomaly detection.

### 5. Held-out evaluation provides a more realistic estimate

The initial experiments used the same sample for model fitting and evaluation.

Later experiments introduced separate training, validation, and test datasets.

The V5 experiment extends this methodology to the complete dataset.

### 6. Anomalies are not necessarily fraud

The final V5 model identifies a portion of fraudulent transactions but also produces a substantial number of false positives.

This demonstrates why anomaly detection should be treated as an investigation tool rather than a definitive fraud classification system.

# 21. Limitations

Finomaly is an academic and exploratory project.

Several limitations should be considered when interpreting its results.

### 1. Synthetic dataset

PaySim is a synthetic dataset.

Its transaction patterns may not fully represent the complexity of real-world financial systems.

### 2. High false-positive rate

The V5 model produces 4,229 false positives for 409 true positives in the held-out test set.

This limits its suitability for direct operational deployment.

### 3. Feature assumptions

Balance-related features may capture patterns specific to the PaySim simulation.

Their usefulness must be independently validated on real financial transaction data.

### 4. Unsupervised learning limitations

Isolation Forest identifies unusual observations, not fraud itself.

A transaction can be unusual without being fraudulent.

Similarly, a fraudulent transaction may not always appear statistically unusual.

### 5. Limited algorithm comparison

The current implementation focuses primarily on Isolation Forest.

Other anomaly-detection algorithms have not been comprehensively benchmarked.

### 6. Interactive evaluation

The dashboard's interactive evaluation uses the same sampled transactions used for the current detection run.

These results are exploratory and must not be interpreted as independent test performance.

The V5 held-out test results provide the primary estimate of model generalisation in this project.

### 7. Not production-ready

The application does not include:

- Real-time transaction streaming.
- Financial institution integrations.
- Automated transaction blocking.
- Human investigation workflows.
- Production model monitoring.
- Deployment-grade authentication and access control.

It is intended for academic demonstration, experimentation, and analysis.

# 22. Future Improvements

Potential directions for extending Finomaly include:

- Comparing Isolation Forest with Local Outlier Factor and One-Class SVM.
- Investigating alternative anomaly-scoring approaches.
- Improving precision while maintaining useful recall.
- Evaluating feature importance and feature contributions.
- Investigating account-level behavioural patterns.
- Testing the model on additional financial datasets.
- Introducing time-aware evaluation.
- Adding model monitoring and experiment comparison.
- Building a real-time transaction analysis pipeline.
- Deploying the application with appropriate security and infrastructure.

These are possible extensions rather than implemented capabilities.

# 23. Contributors

Developed as an academic mini-project.

**Project:** Finomaly — Financial Intelligence & Anomaly Mining Platform

**Repository:**  
https://github.com/amargurung1013/finomaly

---

## Final Note

Finomaly demonstrates an end-to-end exploratory machine learning workflow, from processing a large transaction dataset to engineering behavioural features, detecting anomalies, evaluating model performance, and presenting results through an interactive dashboard.

The project also demonstrates an important principle in machine learning:

**Detecting unusual behaviour is not the same as proving fraudulent behaviour.**

A useful anomaly-detection system must be evaluated not only by how many suspicious transactions it identifies, but also by how reliably it distinguishes genuinely concerning activity from legitimate variation.