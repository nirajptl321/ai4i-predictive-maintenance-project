# MLP and XGBoost Training Notes

Two supervised tabular classifiers were trained on the processed AI4I feature set.

## Inputs

- Features: `type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min`
- Target: `machine_failure`
- Excluded columns: `twf, hdf, pwf, osf, rnf`
- Split: stratified 70/15/15 train/validation/test

## Search setup

- `MLPClassifier`: hidden layer size, alpha, and learning rate
- `XGBoost`: tree count, depth, learning rate, and class weight
- Trial selection metric: `f1_score`
- Threshold selection metric: `f1_score`

## Saved files

- Validation summary: `mlp_xgboost/results/validation_metrics.csv`
- Hyperparameter trials: `mlp_xgboost/results/hyperparameter_trials.csv`
- MLP model: `mlp_xgboost/models/mlp_classifier.joblib`
- XGBoost model: `mlp_xgboost/models/xgboost_classifier.joblib`
- Best model: `mlp_xgboost/models/best_mlp_xgboost_model.joblib`

## Best validation model

`XGBoost` had validation F1 = 0.6429, recall = 0.7059, F2 = 0.6792, and ROC-AUC = 0.9607.

Total validation trials: 12.
