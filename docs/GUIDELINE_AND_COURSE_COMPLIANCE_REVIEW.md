# Guideline and Course Compliance Review

Date: 2026-06-14

This review checks the project against the official COEN 330 project guideline and local course lecture/tutorial materials. It focuses on the technical project work: data processing, leakage prevention, EDA, model comparison, tuning, validation/test separation, final evaluation, demo, reproducibility, and GitHub clarity.

## Sources Used

- Official guideline: `docs/COEN330_Project_Guidelines.pdf`
- Course materials: `docs/course_materials/` was used locally for audit/review and is intentionally excluded from the final Moodle ZIP.
- Course concept summary: `docs/COURSE_CONTENT_AUDIT.md`
- Technical project review: `docs/TECHNICAL_REVIEW.md`
- Model tuning summary: `docs/MODEL_TUNING_SUMMARY.md`

## Compliance Table

| Requirement | Status | Evidence files | Notes | Fix needed |
|---|---|---|---|---|
| Clear problem definition | Complete | `README.md`, `docs/TECHNICAL_REVIEW.md`, `report/final_report.md` | The project is framed as supervised binary classification for machine failure prediction. | None |
| Practical motivation | Complete | `README.md`, `report/final_report.md` | Missed failures are linked to downtime, repair cost, and safety risk. | None |
| Dataset source | Complete | `README.md`, `data/data_link.txt`, `report/final_report.md` | AI4I 2020 Predictive Maintenance Dataset from UCI is documented. | None |
| Dataset description | Complete | `README.md`, `docs/TECHNICAL_REVIEW.md`, `report/final_report.md` | Notes synthetic data, row count, target, feature list, and failure-mode columns. | None |
| Number of rows/features | Complete | `README.md`, `report/final_report.md`, `data/processed/ai4i_processed.csv` | Documents 10,000 rows and the approved six model input features. | None |
| Target definition | Complete | `README.md`, `src/config.py`, `docs/TECHNICAL_REVIEW.md` | `machine_failure`; 0 is no failure and 1 is failure. | None |
| Data types | Complete | `README.md`, `docs/TECHNICAL_REVIEW.md`, `src/config.py` | One categorical feature and five numeric features are documented. | None |
| Missing values | Complete | `results/missing_values_summary.csv`, `src/eda.py`, `report/final_report.md` | Missing-value summary is generated and included in EDA. | None |
| Outlier discussion | Complete | `results/plots/feature_distributions.png`, `docs/TECHNICAL_REVIEW.md`, `report/final_report.md` | Feature distributions support a basic review of unusual values. No automatic row removal is applied because there was no clear preprocessing bug. | None |
| Preprocessing decisions | Complete | `src/preprocessing.py`, `README.md`, `docs/TECHNICAL_REVIEW.md` | Cleans names, drops IDs, preserves diagnostic labels for EDA, and writes processed CSV. | None |
| Feature engineering / feature selection | Complete | `src/config.py`, `src/modeling.py`, `README.md` | Selects six approved features and excludes ID/leakage columns. | None |
| Categorical encoding | Complete | `src/modeling.py`, `README.md` | `type` is one-hot encoded in the model pipeline. | None |
| Numerical scaling | Complete | `src/modeling.py`, `docs/TECHNICAL_REVIEW.md` | Numeric features use `StandardScaler` inside the pipeline. | None |
| Train/validation/test split | Complete | `src/utils.py`, `README.md`, `docs/TECHNICAL_REVIEW.md` | Stratified 70/15/15 split is used. | None |
| Random seed | Complete | `src/config.py`, `src/utils.py`, `src/modeling.py` | `RANDOM_STATE = 42` is used where relevant. | None |
| Leakage prevention | Complete | `src/config.py`, `src/utils.py`, `src/eda.py`, `docs/TECHNICAL_REVIEW.md` | `twf`, `hdf`, `pwf`, `osf`, and `rnf` are not model features. | None |
| EDA plots and observations | Complete | `src/eda.py`, `results/plots/`, `results/missing_values_summary.csv` | Class balance, missing values, distributions, correlation, target-vs-feature, and failure modes are covered. Plot layout and clipping were fixed at the source and visually checked. | None |
| At least five models | Complete | `src/modeling.py`, `results/metrics_table.csv`, `README.md` | Exactly five required models are trained. | None |
| Baseline model | Complete | `src/modeling.py`, `README.md` | Logistic Regression is the simple baseline. | None |
| Hyperparameter tuning for at least three models | Complete | `src/modeling.py`, `results/hyperparameter_trials.csv`, `docs/MODEL_TUNING_SUMMARY.md`, `README.md` | Decision Tree, Random Forest, and HistGradientBoostingClassifier are tuned. The full history has 36 validation trials. | None |
| Model selection using validation, not test | Complete | `src/train.py`, `src/modeling.py`, `results/metrics_table.csv` | Final model is selected by validation F1-score. The test split is not used during selection. `results/metrics_table.csv` contains the best validation result per model. | None |
| Final test evaluation | Complete | `src/evaluate.py`, `results/test_metrics.csv` | Saved final model is evaluated once on the held-out test split. | None |
| Appropriate metrics | Complete | `src/utils.py`, `results/test_metrics.csv`, `README.md` | Accuracy, precision, recall, F1-score, F2-score, ROC-AUC, and confusion matrix are reported. | None |
| Model comparison table | Complete | `results/metrics_table.csv`, `results/plots/model_comparison.png` | Validation results compare all five models. | None |
| Confusion matrix | Complete | `results/test_metrics.csv`, `results/plots/confusion_matrix.png`, `README.md` | Final counts are TN=1444, FP=5, FN=17, TP=34. | None |
| False positive / false negative discussion | Complete | `README.md`, `docs/TECHNICAL_REVIEW.md`, `report/final_report.md` | Explains 5 false alarms and 17 missed failures. | None |
| Feature importance / interpretation | Complete | `src/evaluate.py`, `results/plots/feature_importance.png`, `docs/TECHNICAL_REVIEW.md` | Feature importance is generated when available for the final model pipeline. | None |
| Limitations | Complete | `README.md`, `docs/TECHNICAL_REVIEW.md`, `report/final_report.md` | Notes synthetic data, rare failures, no threshold tuning, and no production deployment. | None |
| Reproducibility | Complete | `README.md`, `requirements.txt`, `results/full_reproducibility_run.txt`, `results/hyperparameter_trials.csv`, `src/` | Commands, saved run output, and full tuning history are present. | None |
| Demo | Complete | `demo/demo.py`, `demo/README.md` | Demo loads the saved model and shows sample predictions. | None |
| Notebooks or scripts | Complete | `notebooks/`, `src/` | Notebooks are walkthroughs; `src/` is the reproducible pipeline. | None |
| Final report sections | Complete | `report/final_report.md`, `report/final_report_draft.md` | All required report sections are present. | None for this technical audit |
| Final report exports | Complete | `report/final_report.pdf`, `report/final_report.docx` | Final PDF and Word report files are present and reviewed for submission. | None |
| Team contribution statement | Complete | `report/final_report.md`, `report/final_report_draft.md` | Team names and factual contribution wording are present. | None |
| References | Complete | `report/final_report.md`, `report/final_report_draft.md` | UCI, scikit-learn, and pandas references are included. | None |
| Academic integrity / external tools acknowledgment | Complete | `README.md`, `report/final_report.md` | tool-supported support and Python libraries are disclosed. | None |
| Final submission checklist | Complete | `docs/final_submission_checklist.md` | Checklist exists and includes remaining manual submission tasks. | None |
| Decision threshold tuning | Partial | `README.md`, `docs/TECHNICAL_REVIEW.md`, `docs/COURSE_CONTENT_AUDIT.md` | Course materials discuss threshold tradeoffs. This project keeps the default threshold and lists tuning as future work. | Optional only |
| PR-AUC or precision-recall curve | Partial | `docs/COURSE_CONTENT_AUDIT.md`, `results/test_metrics.csv` | Guideline mentions PR-AUC for imbalanced classification. The project instead reports recall, F1, F2, ROC-AUC, and confusion matrix. | Optional only |
| Final Moodle ZIP | Complete | `docs/final_submission_checklist.md` | Final ZIP exists and keeps only the project files needed for submission. It intentionally excludes local course materials and development folders. | None |

## Code Correctness Audit

No real code bug was found.

- `src/config.py` defines clear constants for paths, columns, target, split settings, and model names.
- `src/data_loading.py` loads raw and processed CSV files through small helper functions.
- `src/preprocessing.py` cleans column names, drops IDs, and saves `data/processed/ai4i_processed.csv`.
- `src/eda.py` generates the expected EDA summaries and plots.
- `src/modeling.py` defines exactly the five required models and the common preprocessing pipeline.
- `src/train.py` uses the validation set for tuning and selection, then refits the final model on train plus validation.
- `src/evaluate.py` evaluates only the saved final model on the held-out test split.
- `demo/demo.py` loads the saved model and prints sample predictions.
- `src/utils.py` creates the approved feature matrix using only `FEATURE_COLUMNS`.

## Final Model Check

Selected model:

- `HistGradientBoostingClassifier`

Best parameters:

- `learning_rate`: 0.05
- `max_iter`: 100
- `max_leaf_nodes`: 31

Final test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.9853 |
| Precision | 0.8718 |
| Recall | 0.6667 |
| F1-score | 0.7556 |
| F2-score | 0.6996 |
| ROC-AUC | 0.9750 |

Confusion matrix:

- TN = 1444
- FP = 5
- FN = 17
- TP = 34

## Final Verdict

Project status: ready from a technical machine learning standpoint and ready for final course submission.

The only remaining item is instructor-specific confirmation, not missing model work:

- Confirm the instructor's exact expectation for AI-tool disclosure wording.
