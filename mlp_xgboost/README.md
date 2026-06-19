# MLP and XGBoost Scripts

This folder keeps the MLPClassifier and XGBoost experiments separate from the main `src/` pipeline. The scripts reuse the same processed AI4I dataset, feature columns, leakage exclusions, and train/validation/test split as the rest of the repository.

## Install

From the repository root:

```bash
pip install -r requirements.txt
pip install -r mlp_xgboost/requirements-xgboost.txt
```

## Run

Create the processed dataset first if it is missing:

```bash
python -m src.preprocessing
```

Then run the scripts:

```bash
python -m mlp_xgboost.train_mlp_xgboost
python -m mlp_xgboost.evaluate_mlp_xgboost
python -m mlp_xgboost.demo_mlp_xgboost
```

Or run everything:

```bash
python -m mlp_xgboost.run_all_mlp_xgboost
```

## Outputs

```text
mlp_xgboost/models/mlp_classifier.joblib
mlp_xgboost/models/xgboost_classifier.joblib
mlp_xgboost/models/best_mlp_xgboost_model.joblib
mlp_xgboost/results/validation_metrics.csv
mlp_xgboost/results/hyperparameter_trials.csv
mlp_xgboost/results/test_metrics.csv
mlp_xgboost/results/test_predictions.csv
mlp_xgboost/results/error_analysis.md
mlp_xgboost/results/training_notes.md
mlp_xgboost/results/plots/
```
