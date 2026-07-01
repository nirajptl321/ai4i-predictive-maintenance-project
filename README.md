# Machine Failure Prediction Using the AI4I 2020 Predictive Maintenance Dataset

This project looks at whether basic machine sensor readings can be used to predict machine failure.

The goal was to build a clean machine learning workflow from raw data to final evaluation. I kept the project simple and reproducible so the results can be checked from the command line.

Machine failure prediction matters because missed failures can lead to downtime, repair costs, and safety problems. For that reason, this project pays attention to recall and F2-score in addition to the main F1-score.

This was prepared as a COEN 330 course project. The repository is organized so the workflow can be reviewed and rerun.

## Dataset

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Source: UCI Machine Learning Repository
- Raw file: `data/raw/ai4i2020.csv`
- Dataset notes: `data/data_link.txt`
- Course guideline copy: `docs/COEN330_Project_Guidelines.pdf`

The dataset is synthetic and small enough to keep in the repository. It has 10,000 rows and a ready-made binary target. I chose AI4I because it fits the course topic well: it is about machine condition data, has a clear failure label, and is small enough for teammates or the instructor to rerun quickly.

Dataset attribution: the AI4I 2020 Predictive Maintenance Dataset is distributed through the UCI Machine Learning Repository under CC BY 4.0. The dataset is credited to Stephan Matzka and is associated with the paper "Explainable Artificial Intelligence for Predictive Maintenance Applications" (2020).

Target:

- `machine_failure = 0`: no machine failure
- `machine_failure = 1`: machine failure

## Features Used

The model uses these input features:

- `type`
- `air_temperature_k`
- `process_temperature_k`
- `rotational_speed_rpm`
- `torque_nm`
- `tool_wear_min`

The original column names are cleaned into Python-friendly snake_case during preprocessing.

ID columns are dropped:

- `UDI`
- `Product ID`

These failure mode columns are not used as model inputs because they leak target information:

- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

They are only used in EDA to explain failure mode counts.

## Project Workflow

1. Load `data/raw/ai4i2020.csv`.
2. Clean column names.
3. Drop ID columns.
4. Keep failure mode columns only for EDA, not training.
5. Use `machine_failure` as the target.
6. One-hot encode `type`.
7. Split the data with stratification and `random_state=42`:
   - 70% train
   - 15% validation
   - 15% test
8. Use the validation set for model selection and tuning.
9. Use the test set once for final evaluation.

## Models Used

Five models were trained and compared:

| Model                          | Purpose                        | Tuning         |
| ------------------------------ | ------------------------------ | -------------- |
| Logistic Regression            | Simple baseline                | Not grid-tuned |
| Decision Tree                  | Interpretable nonlinear model  | Tuned          |
| Random Forest                  | Ensemble tree model            | Tuned          |
| Extra Trees                    | Randomized ensemble tree model | Not grid-tuned |
| HistGradientBoostingClassifier | Final selected boosting model  | Tuned          |

The tuned models used small grids so the project would run quickly:

- Decision Tree: `criterion`, `max_depth`, `min_samples_leaf`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`
- HistGradientBoostingClassifier: `learning_rate`, `max_iter`, `max_leaf_nodes`

`class_weight="balanced"` is used for Logistic Regression, Decision Tree, Random Forest, and Extra Trees.

The full tuning history is saved for reproducibility. The ranked best validation result per model is saved in `results/metrics_table.csv`, and all 36 validation trials are saved in `results/hyperparameter_trials.csv`. Only the final selected model object is saved in `models/final_model.joblib`.

## Results

Selected model:

`HistGradientBoostingClassifier`

Test metrics:

- Accuracy: 0.9853
- Precision: 0.8718
- Recall: 0.6667
- F1-score: 0.7556
- F2-score: 0.6996
- ROC-AUC: 0.9750
- Confusion matrix: TN=1444, FP=5, FN=17, TP=34

The model caught 34 failure cases and missed 17 failure cases in the test set. It also produced 5 false alarms. Accuracy is high, but recall is important because missed failures are costly.

Best validation results by model are in `results/metrics_table.csv`. The complete validation tuning history is in `results/hyperparameter_trials.csv`. Final test results are in `results/test_metrics.csv`.

## How To Run

Recommended Python version: Python 3.11 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python -m src.preprocessing
python -m src.eda
python -m src.train
python -m src.evaluate
python demo/demo.py
```

Capture the full run output:

```bash
{
  python -m src.preprocessing
  python -m src.eda
  python -m src.train
  python -m src.evaluate
  python demo/demo.py
} > results/full_reproducibility_run.txt 2>&1
```

Run only the demo:

```bash
python demo/demo.py
```

The demo loads `models/final_model.joblib`, reads sample rows from `data/processed/ai4i_processed.csv`, and prints the true class, predicted class, and predicted probability of machine failure.

This is only a local course-project demo. It is not a deployment-ready maintenance system.

## Where Things Are

- Source code: `src/`
- Demo: `demo/`
- Notebooks: `notebooks/`
- Results and plots: `results/`
- Final model: `models/final_model.joblib`
- Final PDF report: `report/final_report.pdf`

The notebooks are walkthroughs for reading and presenting the work. The reproducible pipeline is in `src/`.

## Expected Outputs

- `data/processed/ai4i_processed.csv`
- `models/final_model.joblib`
- `results/metrics_table.csv`
- `results/hyperparameter_trials.csv`
- `results/test_metrics.csv`
- `results/missing_values_summary.csv`
- `results/full_reproducibility_run.txt`
- `results/demo_output.txt`
- `results/plots/class_balance.png`
- `results/plots/confusion_matrix.png`
- `results/plots/model_comparison.png`
- `results/plots/feature_importance.png`
- `results/plots/failure_mode_counts.png`
- `results/plots/target_vs_features.png`
- `report/final_report.pdf`

The plot files in `results/plots/` are generated from the plotting code. Text clipping and layout issues were fixed at the source, and the regenerated PNGs were visually checked.

## Limitations

- The dataset is synthetic, so the results may not transfer directly to real industrial machines.
- The failure class is rare, so recall and F2-score matter more than accuracy alone.
- The project does not tune the decision threshold after selecting the final model.
- The demo is local and simple. It is not deployment software.
- The project does not include live monitoring, retraining, or production data ingestion.

## Academic Integrity / External Tools Note

This repository is set up so the pipeline can be rerun and checked. Python libraries used include pandas, scikit-learn, matplotlib, seaborn, joblib, and nbformat. The team reviewed and verified the final repository contents and is responsible for the submitted work.
