# Comprehensive Project Review

## Guideline PDF Status

Copied successfully: yes.

Guideline destination:

- `docs/COEN330_Project_Guidelines.pdf`

Guideline source used:

- `/home/niraj/Downloads/COEN330_Project_Guidelines_Extended_Complete_Summer2026.pdf`

PDF details observed with `pdfinfo`:

- Title/content: COEN 330 Applied Machine Learning course project guidelines, Summer 2026
- Pages: 6
- File size: 404,687 bytes

## GitHub Readiness Check

`.gitignore` was inspected. It keeps the required AI4I project artifacts commit-visible and continues to ignore development noise.

Commit-visible or already tracked:

- `data/raw/ai4i2020.csv` - already tracked by git
- `data/processed/ai4i_processed.csv` - visible as untracked
- `models/final_model.joblib` - visible as untracked

Ignored as intended:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `*.tmp`
- `*.log`
- `.DS_Store`
- `Thumbs.db`
- `data/raw/*.zip`

Git LFS status: not used and not needed for the small AI4I CSV/model artifacts.

## Guideline Items Checked

The compliance check used:

- `docs/COEN330_Project_Guidelines.pdf`
- `docs/guideline_checklist.md`
- `docs/final_submission_checklist.md`
- `report/final_report_draft.md`
- `report/final_report.md`
- `README.md`
- `PROJECT_LOG.md`
- `results/`
- `src/`
- `demo/`
- `notebooks/`

## Passed

- Clear problem definition: supervised binary machine failure classification.
- Dataset source and description: AI4I 2020 Predictive Maintenance Dataset from UCI.
- Target definition: `machine_failure`, with 0 as no failure and 1 as failure.
- Preprocessing: raw CSV loading, column-name cleaning, ID removal, processed CSV output.
- EDA: class balance, missing values, feature distributions, correlation heatmap, target-vs-feature plots, failure mode counts.
- Feature engineering and representation: feature selection, one-hot encoding for `type`, numeric standardization in the modeling pipeline.
- At least five models: exactly five models are implemented and compared.
- Simple baseline model: Logistic Regression baseline.
- Hyperparameter tuning for at least three models: Decision Tree, Random Forest, and HistGradientBoostingClassifier.
- Validation/test separation: stratified 70/15/15 split with validation for selection and test for final evaluation.
- Target leakage control: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from model features.
- Appropriate metrics: accuracy, precision, recall, F1-score, F2-score, ROC-AUC, and confusion matrix.
- Model comparison table: `results/metrics_table.csv` and report section 9.
- Error analysis: false positives/false negatives and recall tradeoff discussed.
- Limitations: synthetic dataset, class imbalance, threshold tuning, and deployment limitations documented.
- Reproducibility instructions: README and `results/full_reproducibility_run.txt`.
- Demo: `demo/demo.py` loads the saved model and predicts on sample processed rows.
- Notebooks: five explanatory notebooks are present under `notebooks/`.
- Final report structure: all 15 required report sections are present.
- Team contribution section: present with placeholder names.
- References: UCI dataset, scikit-learn, and pandas references are included.
- Academic integrity / AI-tool acknowledgment: README and report now include external tools and AI-assistance disclosure text.

## Five Models Verified

Documented and implemented:

- Logistic Regression baseline
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoostingClassifier

Evidence:

- `src/modeling.py`
- `README.md`
- `report/final_report.md`
- `report/final_report_draft.md`
- `results/metrics_table.csv`

## Hyperparameter Tuning Verified

Tuned models:

- Decision Tree
- Random Forest
- HistGradientBoostingClassifier

Documented tuning fields:

- Decision Tree: `criterion`, `max_depth`, `min_samples_leaf`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`
- HistGradientBoostingClassifier: `learning_rate`, `max_iter`, `max_leaf_nodes`

Evidence:

- `src/modeling.py`
- `README.md`
- `report/final_report.md`
- `report/final_report_draft.md`
- `results/metrics_table.csv`

## Final Model and Test Metrics Verified

Selected model:

- HistGradientBoostingClassifier

Held-out test metrics:

- Accuracy: 0.9853
- Precision: 0.8718
- Recall: 0.6667
- F1-score: 0.7556
- F2-score: 0.6996
- ROC-AUC: 0.9750
- Confusion matrix: TN=1444, FP=5, FN=17, TP=34

Evidence:

- `results/test_metrics.csv`
- `results/full_reproducibility_run.txt`
- `README.md`
- `report/final_report.md`
- `report/final_report_draft.md`
- `models/final_model.joblib`

## Partial Items

- Team contribution names are placeholders (`TODO_NAME`) until the team fills in real names.
- The guideline asks for final report submission as a PDF for Moodle. The repository currently includes Markdown report files: `report/final_report.md` and `report/final_report_draft.md`. Export to PDF remains a final packaging step if Moodle requires it.
- The guideline recommends screenshots or a short video for the demo. The project includes a working command-line demo and captured `results/demo_output.txt`, but no screenshot or video.

## Missing Required Items

No required repository component was found missing during this review.

## Evidence File Paths

- Official guideline PDF: `docs/COEN330_Project_Guidelines.pdf`
- README: `README.md`
- Project log: `PROJECT_LOG.md`
- Guideline checklist: `docs/guideline_checklist.md`
- Final submission checklist: `docs/final_submission_checklist.md`
- Final report draft: `report/final_report_draft.md`
- Final report: `report/final_report.md`
- Report notes: `report/REPORT_NOTES.md`
- Preprocessing source: `src/preprocessing.py`
- EDA source: `src/eda.py`
- Modeling source: `src/modeling.py`
- Training source: `src/train.py`
- Evaluation source: `src/evaluate.py`
- Demo source: `demo/demo.py`
- Processed dataset: `data/processed/ai4i_processed.csv`
- Final model: `models/final_model.joblib`
- Validation metrics: `results/metrics_table.csv`
- Test metrics: `results/test_metrics.csv`
- Reproducibility log: `results/full_reproducibility_run.txt`
- Demo output: `results/demo_output.txt`
- Plots: `results/plots/`
- Notebooks: `notebooks/`

## Final Remaining TODOs

- Replace `TODO_NAME` placeholders with real team member names.
- Export `report/final_report.md` to PDF if required for Moodle submission.
- Confirm the final external tools and AI-assistance disclosure wording matches the instructor's policy.
- Optionally add a demo screenshot or short video if the team wants to follow the recommendation in the guideline PDF.
- Commit the repository intentionally after reviewing all untracked files.

