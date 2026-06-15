# Model Tuning Summary

This project trains exactly five models on the AI4I processed feature set:

| Model | Tuned? | Validation trials |
|---|---:|---:|
| Logistic Regression | No | 1 |
| Decision Tree | Yes | 18 |
| Random Forest | Yes | 8 |
| Extra Trees | No | 1 |
| HistGradientBoostingClassifier | Yes | 8 |

Total validation trials saved: 36.

## Tuning Grids

Logistic Regression is used as a fixed baseline and is not grid-tuned. Its estimator settings are `class_weight="balanced"`, `max_iter=1000`, `random_state=42`, and `solver="liblinear"`.

Decision Tree is tuned with this grid:

| Hyperparameter | Values |
|---|---|
| `criterion` | `"gini"`, `"entropy"` |
| `max_depth` | `3`, `5`, `None` |
| `min_samples_leaf` | `1`, `5`, `10` |

Random Forest is tuned with this grid:

| Hyperparameter | Values |
|---|---|
| `n_estimators` | `100`, `200` |
| `max_depth` | `8`, `None` |
| `min_samples_leaf` | `1`, `5` |

Extra Trees is used as a fixed ensemble baseline and is not grid-tuned. Its estimator settings are `class_weight="balanced"`, `n_estimators=200`, `n_jobs=-1`, and `random_state=42`.

HistGradientBoostingClassifier is tuned with this grid:

| Hyperparameter | Values |
|---|---|
| `learning_rate` | `0.05`, `0.10` |
| `max_iter` | `100`, `200` |
| `max_leaf_nodes` | `15`, `31` |

## Selection and Saved Outputs

The split is stratified with `random_state=42`: 70% train, 15% validation, and 15% test.

All candidate hyperparameter combinations are fit on the training split and evaluated on the validation split. The validation set is used for model selection, with the final model selected by validation F1-score using the existing project ranking rule. The test set is held back and used only for final held-out evaluation after the final model is selected and refit on train plus validation data.

The ranked best validation result per model is saved in `results/metrics_table.csv`. The complete validation trial history for every model and every tested hyperparameter combination is saved in `results/hyperparameter_trials.csv`.

Only the final selected model object is saved to `models/final_model.joblib`. Candidate model objects from individual tuning trials are not saved.
