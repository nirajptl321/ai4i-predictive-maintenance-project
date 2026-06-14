# Technical Review

This review focuses on the working machine learning project, not the final report formatting. It checks the parts that matter for defending the project: data processing, leakage prevention, EDA, model comparison, tuning, evaluation, demo, and reproducibility.

## Project Task

- Project: Machine Failure Prediction Using the AI4I 2020 Predictive Maintenance Dataset
- Task type: supervised binary classification
- Target: `machine_failure`
- Target meaning:
  - `0`: no machine failure
  - `1`: machine failure

The project asks whether basic machine sensor readings can predict the binary machine failure target.

## Dataset

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Source: UCI Machine Learning Repository
- Raw file: `data/raw/ai4i2020.csv`
- Processed file: `data/processed/ai4i_processed.csv`
- Dataset notes: `data/data_link.txt`

The dataset is synthetic and small enough to commit directly to the repository.

## Preprocessing

Main file:

- `src/preprocessing.py`

What preprocessing does:

- Loads `data/raw/ai4i2020.csv`.
- Cleans original column names into snake_case.
- Drops ID columns if present:
  - `udi`
  - `product_id`
- Saves the processed CSV to `data/processed/ai4i_processed.csv`.
- Keeps failure mode columns in the processed file for EDA/explanation only.

Supporting files:

- `src/config.py`: central constants, paths, feature names, target name, and leakage columns.
- `src/data_loading.py`: raw and processed CSV loading helpers.
- `src/utils.py`: shared split and metrics helpers.

## Model Feature Columns

The approved model features are defined in `src/config.py` as `FEATURE_COLUMNS`:

- `type`
- `air_temperature_k`
- `process_temperature_k`
- `rotational_speed_rpm`
- `torque_nm`
- `tool_wear_min`

The model matrix is created in `src/utils.py` by `split_features_target()`, which uses only `FEATURE_COLUMNS` and `TARGET_COLUMN`.

## Leakage Prevention

These diagnostic failure mode columns are excluded from model features:

- `twf`
- `hdf`
- `pwf`
- `osf`
- `rnf`

Evidence:

- `src/config.py` lists them in `LEAKAGE_COLUMNS`, not `FEATURE_COLUMNS`.
- `src/utils.py` creates `X = df[FEATURE_COLUMNS].copy()`.
- `src/utils.py` checks that no leakage column is included in `FEATURE_COLUMNS`.
- `src/eda.py` uses `LEAKAGE_COLUMNS` only for the failure mode count plot.
- The saved model package lists `excluded_leakage_columns`: `['twf', 'hdf', 'pwf', 'osf', 'rnf']`.

Conclusion: the failure mode columns are used for EDA/explanation only, not for model training or evaluation.

## EDA

Main file:

- `src/eda.py`

Generated EDA outputs:

- `results/missing_values_summary.csv`
- `results/plots/class_balance.png`
- `results/plots/feature_distributions.png`
- `results/plots/correlation_heatmap.png`
- `results/plots/target_vs_features.png`
- `results/plots/failure_mode_counts.png`

The failure mode count plot is explanatory only and is not part of the model feature set.

## Train / Validation / Test Split

Split helper:

- `src/utils.py`: `make_train_validation_test_split()`

Split settings:

- Train: 70%
- Validation: 15%
- Test: 15%
- Stratification: yes, on `machine_failure`
- Random seed: `random_state=42`

The validation set is used for model selection and tuning. The test set is used for final evaluation after the final model is selected.

## Five Models

The five models are defined in `src/modeling.py` inside `model_specs()`:

| Model | Role | Tuning |
|---|---|---|
| Logistic Regression | Simple baseline | Not grid-tuned |
| Decision Tree | Interpretable nonlinear model | Tuned |
| Random Forest | Ensemble tree model | Tuned |
| Extra Trees | Randomized ensemble tree model | Not grid-tuned |
| HistGradientBoostingClassifier | Boosting model and final selected model | Tuned |

Tuned models:

- Decision Tree
- Random Forest
- HistGradientBoostingClassifier

Validation results are saved in:

- `results/metrics_table.csv`

## Final Selected Model

Selected model:

- HistGradientBoostingClassifier

Best parameters:

- `learning_rate`: 0.05
- `max_iter`: 100
- `max_leaf_nodes`: 31

Evidence:

- `results/metrics_table.csv`
- `models/final_model.joblib`
- `results/full_reproducibility_run.txt`

## Final Test Metrics

Final test metrics are saved in `results/test_metrics.csv`.

| Metric | Value |
|---|---:|
| Accuracy | 0.9853 |
| Precision | 0.8718 |
| Recall | 0.6667 |
| F1-score | 0.7556 |
| F2-score | 0.6996 |
| ROC-AUC | 0.9750 |

Confusion matrix counts:

- True negatives: 1444
- False positives: 5
- False negatives: 17
- True positives: 34

Plain-language interpretation: the model caught 34 failure cases and missed 17 failure cases in the test set. It also produced 5 false alarms. The accuracy is high, but recall and F2-score matter because missed failures are costly.

## Demo Behavior

Demo file:

- `demo/demo.py`

The demo:

- Loads `models/final_model.joblib`.
- Loads sample rows from `data/processed/ai4i_processed.csv`.
- Prints true class, predicted class, and predicted probability of machine failure.
- States that it is a local demo, not deployment software.

Captured demo output:

- `results/demo_output.txt`

## Reproducibility Commands

Run the full pipeline from the repository root:

```bash
python -m src.preprocessing
python -m src.eda
python -m src.train
python -m src.evaluate
python demo/demo.py
```

Run only the demo:

```bash
python demo/demo.py
```

Check Python syntax/import compilation:

```bash
python -m compileall -q src demo
```

## Source Organization Review

- `src/config.py`: clear central constants for paths, columns, splits, target, and model names.
- `src/data_loading.py`: clear raw and processed CSV loading helpers.
- `src/preprocessing.py`: clear preprocessing flow for column cleaning, ID removal, and processed CSV output.
- `src/eda.py`: creates all expected EDA plots and summaries.
- `src/modeling.py`: clearly defines the five model specifications, preprocessing pipeline, tuning grids, and validation selection logic.
- `src/train.py`: trains, tunes, selects the model by validation F1-score, and saves the final model package.
- `src/evaluate.py`: loads the saved final model and evaluates it on the held-out test split.
- `src/utils.py`: contains shared split, metric, leakage validation, and confusion matrix helpers.

## Remaining Technical Issues

No blocking technical issues were found in this review.

Known modeling limitations remain:

- The dataset is synthetic.
- The failure class is rare.
- The final model missed 17 failures on the test set.
- The project does not tune the probability threshold.
- The demo is local and is not a production maintenance system.

