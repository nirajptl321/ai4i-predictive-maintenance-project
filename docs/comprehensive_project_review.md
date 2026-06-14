# Comprehensive Project Review

This review checks the AI4I predictive maintenance project against the official COEN 330 project guideline PDF and the current repository files.

## Guideline Source Used

The official guideline PDF exists and was used for this review:

- `docs/COEN330_Project_Guidelines.pdf`

The PDF is the COEN 330 Applied Machine Learning Summer 2026 project guideline. It asks for a clear machine learning problem, dataset documentation, preprocessing, EDA, feature engineering or feature selection, model development, validation, evaluation, interpretation, reproducibility, a demo, a final report, and a team contribution section.

## Guideline Compliance Table

| Requirement | Status | Evidence files | Notes / fixes needed |
|---|---|---|---|
| Problem definition | Complete | `README.md`, `report/final_report.md` | The project is clearly framed as supervised binary classification for machine failure prediction. |
| Motivation and application context | Complete | `README.md`, `report/final_report.md` | The docs explain why missed machine failures matter. |
| Dataset source | Complete | `README.md`, `data/data_link.txt`, `report/final_report.md` | Dataset is AI4I 2020 Predictive Maintenance Dataset from UCI. |
| Dataset description | Complete | `README.md`, `report/final_report.md`, `data/processed/ai4i_processed.csv` | Documents 10,000 synthetic observations, target, feature types, and approved inputs. |
| Target definition | Complete | `README.md`, `report/final_report.md`, `src/config.py` | `machine_failure`; 0 means no failure and 1 means failure. |
| Preprocessing | Complete | `src/preprocessing.py`, `README.md`, `report/final_report.md` | Cleans column names, drops ID columns, and writes `data/processed/ai4i_processed.csv`. |
| EDA | Complete | `src/eda.py`, `results/plots/`, `results/missing_values_summary.csv`, `report/final_report.md` | Includes class balance, missing values, feature distributions, correlation heatmap, target-vs-feature plots, and failure mode counts. |
| Feature engineering / representation | Complete | `src/modeling.py`, `README.md`, `report/final_report.md` | Uses one-hot encoding for `type`, numeric standardization, and feature selection. |
| Leakage prevention | Complete | `src/config.py`, `src/utils.py`, `README.md`, `report/final_report.md` | `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from model features and used only for EDA/explanation. |
| Train/validation/test split | Complete | `src/utils.py`, `README.md`, `report/final_report.md` | Stratified 70/15/15 split with `random_state=42`. |
| Five models | Complete | `src/modeling.py`, `results/metrics_table.csv`, `README.md` | Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoostingClassifier. |
| Baseline model | Complete | `src/modeling.py`, `README.md`, `report/final_report.md` | Logistic Regression is the simple baseline. |
| Hyperparameter tuning for at least three models | Complete | `src/modeling.py`, `README.md`, `results/metrics_table.csv` | Decision Tree, Random Forest, and HistGradientBoostingClassifier are tuned. |
| Validation / model selection | Complete | `src/train.py`, `src/modeling.py`, `results/metrics_table.csv` | Final model is selected by validation F1-score. |
| Final test evaluation | Complete | `src/evaluate.py`, `results/test_metrics.csv`, `report/final_report.md` | Test set is used for final evaluation after model selection. |
| Appropriate metrics | Complete | `src/utils.py`, `results/test_metrics.csv`, `README.md` | Accuracy, precision, recall, F1-score, F2-score, ROC-AUC, and confusion matrix are reported. |
| Model comparison | Complete | `results/metrics_table.csv`, `results/plots/model_comparison.png`, `report/final_report.md` | Validation comparison table and plot are available. |
| Error analysis | Complete | `report/final_report.md`, `README.md` | Explains false positives, false negatives, missed failures, and recall tradeoff. |
| Limitations | Complete | `README.md`, `report/final_report.md` | Notes synthetic data, rare failure class, no threshold tuning, and no production deployment. |
| Reproducibility | Complete | `README.md`, `requirements.txt`, `results/full_reproducibility_run.txt`, `src/` | Commands and dependency file are provided. |
| Demo | Complete | `demo/demo.py`, `demo/README.md`, `results/demo_output.txt` | Local command-line demo loads the trained model and predicts on sample rows. |
| Notebooks or scripts | Complete | `notebooks/`, `src/` | Notebooks are walkthroughs; `src/` is the authoritative pipeline. |
| Saved model / results / figures | Complete | `models/final_model.joblib`, `results/`, `results/plots/` | Final model, metrics, logs, and plots are present. |
| Final report structure | Complete | `report/final_report.md`, `report/final_report_draft.md` | All 15 required sections are present. |
| Team contribution section | Complete | `report/final_report.md`, `report/final_report_draft.md` | Team names and contribution wording are finalized. |
| References | Complete | `report/final_report.md`, `report/final_report_draft.md` | Includes UCI, scikit-learn, and pandas references. |
| Academic integrity / external tools acknowledgment | Complete | `README.md`, `report/final_report.md` | External libraries and tool-supported support are acknowledged. |
| Final submission checklist | Complete | `docs/final_submission_checklist.md` | Checklist exists and includes remaining manual tasks. |
| Final report PDF | Partial | `report/final_report.pdf`, `docs/final_submission_checklist.md` | PDF exists, but it should be regenerated after any final report Markdown edit and visually inspected before submission. |
| Moodle ZIP packaging | Partial | `docs/final_submission_checklist.md` | Final ZIP must still be created manually before Moodle submission. |

## Report Structure Check

Both `report/final_report.md` and `report/final_report_draft.md` include the required sections:

1. Abstract
2. Introduction and motivation
3. Related work or background
4. Dataset description
5. Preprocessing and exploratory data analysis
6. Methodology and models
7. Validation and hyperparameter tuning strategy
8. Experimental setup
9. Results and model comparison
10. Error analysis and qualitative discussion
11. Demo or usage demonstration description
12. Limitations and future work
13. Conclusion
14. Team contribution section
15. References

## Final Model and Test Metrics

Selected model:

- HistGradientBoostingClassifier

Final test metrics:

- Accuracy: 0.9853
- Precision: 0.8718
- Recall: 0.6667
- F1-score: 0.7556
- F2-score: 0.6996
- ROC-AUC: 0.9750
- Confusion matrix: TN=1444, FP=5, FN=17, TP=34

Plain-language interpretation: the model caught 34 failure cases and missed 17 failure cases in the test set. It also produced 5 false alarms. Accuracy is high, but recall is important because missed failures are costly.

## Final Project Verdict

Verdict: almost ready.

The project satisfies the main COEN 330 technical and documentation requirements. The remaining items are manual submission tasks rather than missing machine learning work.

## Remaining TODOs

- Team contribution names have been filled in; do a final human review before submission.
- Regenerate `report/final_report.pdf` if `report/final_report.md` changes again.
- Visually inspect `report/final_report.pdf` before submission.
- Create the final Moodle ZIP.
- Confirm the external tools and AI-assistance disclosure wording matches the instructor's policy.
- Optionally add a demo screenshot or short video if the team wants to follow the guideline recommendation.
