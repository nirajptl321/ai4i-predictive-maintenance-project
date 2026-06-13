# Machine Failure Prediction Using the AI4I 2020 Predictive Maintenance Dataset

## Problem Statement

This COEN 330 Applied Machine Learning project predicts whether a machine will fail using the AI4I 2020 Predictive Maintenance Dataset. The task is supervised binary classification.

Target:

- `machine_failure = 0`: no machine failure
- `machine_failure = 1`: machine failure

The main metric is F1-score. Recall and F2-score are secondary metrics because missing a failure is costly.

## Dataset Source

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Source: UCI Machine Learning Repository
- Local file: `data/raw/ai4i2020.csv`
- Link notes: `data/data_link.txt`
- Course guideline copy: `docs/COEN330_Project_Guidelines.pdf`

The dataset is small and synthetic, with 10,000 rows and a ready-made machine failure target.

## GitHub Data Policy

The AI4I CSV files and final model are small enough to commit directly to GitHub:

- Commit `data/raw/ai4i2020.csv`
- Commit `data/processed/ai4i_processed.csv`
- Commit `models/final_model.joblib`

Do not use Git LFS for this project unless a future file becomes large. Downloaded ZIP files such as `data/raw/*.zip` remain ignored.

## Input Features

Only these model input features are used:

- `type`
- `air_temperature_k`
- `process_temperature_k`
- `rotational_speed_rpm`
- `torque_nm`
- `tool_wear_min`

The preprocessing step converts the original column names into Python-friendly names.

## Excluded Columns

ID columns are dropped if present:

- `UDI`
- `Product ID`

Failure mode columns are not used as model features because they leak target information:

- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

These columns are kept only for EDA explanation, especially the failure mode count plot.

## Methodology

1. Load `data/raw/ai4i2020.csv`.
2. Clean column names into snake_case.
3. Drop ID columns.
4. Keep diagnostic failure mode columns only for EDA, not modeling.
5. Use `machine_failure` as the target.
6. One-hot encode `type`.
7. Split data with `random_state=42` and stratification:
   - 70% train
   - 15% validation
   - 15% test
8. Tune/select models using the validation set.
9. Use the test set only once for final evaluation.

## Feature Engineering and Representation

- Column names are cleaned into Python-friendly snake_case.
- `type` is one-hot encoded.
- Numeric features are standardized inside the modeling pipeline.
- Feature selection removes ID columns and excludes diagnostic failure mode columns to prevent target leakage.
- No extra synthetic features are created; the project intentionally keeps the representation simple and reproducible.

## Models

Exactly five models are trained and compared:

- Logistic Regression baseline
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoostingClassifier

Hyperparameter tuning is applied to:

- Decision Tree
- Random Forest
- HistGradientBoostingClassifier

Small grids are used so the full pipeline runs quickly. `class_weight="balanced"` is used for Logistic Regression, Decision Tree, Random Forest, and Extra Trees.

Tuning grids:

- Decision Tree: `criterion`, `max_depth`, `min_samples_leaf`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`
- HistGradientBoostingClassifier: `learning_rate`, `max_iter`, `max_leaf_nodes`

## Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC
- Confusion matrix

## Current Results

Validation selected `HistGradientBoostingClassifier` using F1-score.

Held-out test metrics from `results/test_metrics.csv`:

- Accuracy: 0.9853
- Precision: 0.8718
- Recall: 0.6667
- F1-score: 0.7556
- F2-score: 0.6996
- ROC-AUC: 0.9750
- Confusion matrix counts: TN=1444, FP=5, FN=17, TP=34

## Install Dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `.venv` already exists, activate it and install the requirements.

## Run the Full Pipeline

From the repository root with the environment activated:

```bash
python -m src.preprocessing
python -m src.eda
python -m src.train
python -m src.evaluate
python demo/demo.py
```

To capture a reproducibility log:

```bash
{
  python -m src.preprocessing
  python -m src.eda
  python -m src.train
  python -m src.evaluate
  python demo/demo.py
} > results/full_reproducibility_run.txt 2>&1
```

To capture only the demo output:

```bash
python demo/demo.py > results/demo_output.txt
```

## Expected Outputs

- `data/processed/ai4i_processed.csv`
- `models/final_model.joblib`
- `results/metrics_table.csv`
- `results/test_metrics.csv`
- `results/missing_values_summary.csv`
- `results/full_reproducibility_run.txt`
- `results/demo_output.txt`
- `results/plots/class_balance.png`
- `results/plots/correlation_heatmap.png`
- `results/plots/feature_distributions.png`
- `results/plots/confusion_matrix.png`
- `results/plots/model_comparison.png`
- `results/plots/feature_importance.png`
- `results/plots/failure_mode_counts.png`
- `results/plots/target_vs_features.png`

## Demo

Run:

```bash
python demo/demo.py
```

The demo loads `models/final_model.joblib`, reads examples from `data/processed/ai4i_processed.csv`, and prints the true class, predicted class, and predicted probability of machine failure. It is a simple local demonstration, not deployment software.

## Limitations

- The dataset is synthetic, so results may not transfer directly to real industrial equipment.
- The positive class is rare, which makes recall and F2-score important.
- The model uses only the approved sensor/product features and does not use failure mode labels for prediction.
- The project does not include live data ingestion, monitoring, deployment, or cost-sensitive threshold optimization.

## Academic Integrity / External Tools Note

This repository is structured so the full pipeline can be rerun and inspected. Python libraries used include pandas, scikit-learn, matplotlib, seaborn, joblib, and nbformat. AI-assisted coding and documentation support was used to help scaffold and review the repository; the team is responsible for understanding, verifying, and disclosing this use according to the course policy. The authoritative implementation is in `src/`, and generated outputs can be reproduced from the raw CSV.
