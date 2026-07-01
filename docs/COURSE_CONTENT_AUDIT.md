# Course Content Audit

Date: 2026-06-14

This audit compares the AI4I predictive maintenance project with the official COEN 330 project guideline and local course lecture/tutorial materials stored in `docs/course_materials/`. Those course materials were used only for local audit/review and are intentionally excluded from the final Moodle ZIP.

The goal was not to add unnecessary features. The goal was to check whether the project matches the course concepts and to identify any weak areas that should be explained before submission.

## Sources Read

- `docs/COEN330_Project_Guidelines.pdf`
- `docs/course_materials/coen330/coen330/course-overview.pdf`
- `docs/course_materials/coen330/coen330/Topic1-Intorduction.pdf`
- `docs/course_materials/coen330/coen330/Topic2-Data.pdf`
- `docs/course_materials/coen330/coen330/Topic3-Supervised Learning I.pdf`
- `docs/course_materials/coen330/coen330/Topic3-Supervised Learning II-V3.pdf`
- `docs/course_materials/coen330/coen330/Topic6-Supervised Learning III k-NN, SVM.pdf`
- `docs/course_materials/coen330/coen330/Topic7-Decision_Trees_Interpretability_Ensembles.pdf`
- `docs/course_materials/coen330/coen330/Lecture 5 Probabilistic Learning Naive Bayes-DRAFT3.pdf`
- `docs/course_materials/coen330/coen330/Lecture_8_Ensemble_Learning_pw_v2.pdf`
- `docs/course_materials/coen330/coen330/Lecture_9_Unsupervised_Learning-V3.pdf`
- `docs/course_materials/coen330/coen330/Tutorial 1.pdf`
- `docs/course_materials/coen330/coen330/Tutorial 2.pdf`
- `docs/course_materials/coen330/coen330/Tutorial 3.pdf`
- `docs/course_materials/coen330/coen330/Tutorial 3 - Solution.pdf`
- `docs/course_materials/coen330/coen330/Tutorial 4.pdf`
- `docs/course_materials/coen330/coen330/COEN 330 Tutorial 5-V2.pdf`
- `docs/course_materials/coen330/coen330/COEN330_Tutorial3_Linear_Regression_Solution.ipynb`
- `docs/course_materials/coen330/coen330/COEN_330_Tutorial_5_Notebook.ipynb`

Zip files in the course materials folder were noted but not treated as primary readable course notes. The final ZIP keeps only the project files needed for submission and excludes `docs/course_materials/`.

## Course Concepts Checked

| Course concept | Where it appears in the course material | How this project uses it | Status |
|---|---|---|---|
| Supervised learning | `course-overview.pdf`, `Topic1-Intorduction.pdf`, `Topic3-Supervised Learning I.pdf`, `Tutorial 1.pdf` | Uses labeled rows from AI4I to learn `machine_failure`. | OK |
| Classification | `Topic1-Intorduction.pdf`, `Topic3-Supervised Learning II-V3.pdf`, `Tutorial 1.pdf` | Frames the task as binary classification: failure vs no failure. | OK |
| Train/validation/test split | `Topic2-Data.pdf`, `Topic3-Supervised Learning I.pdf`, `Topic3-Supervised Learning II-V3.pdf`, guideline PDF | Uses a stratified 70/15/15 split with `random_state=42`. | OK |
| Cross-validation or validation strategy | `Topic3-Supervised Learning II-V3.pdf`, guideline PDF | Uses a fixed validation set for model selection and tuning. Cross-validation is discussed by the course but is not required here because the guideline allows a train/validation/test split. | OK |
| Preprocessing | `Topic2-Data.pdf`, `Tutorial 2.pdf`, guideline PDF | Cleans column names, drops ID columns, preserves diagnostic labels for EDA, and saves a processed CSV. | OK |
| Feature scaling | `Topic2-Data.pdf`, `Tutorial 2.pdf`, `COEN_330_Tutorial_5_Notebook.ipynb` | Uses `StandardScaler` for numeric features inside the model pipeline. | OK |
| Categorical encoding | `Topic2-Data.pdf`, `Tutorial 2.pdf`, guideline PDF | Uses one-hot encoding for `type`. | OK |
| Class imbalance | `Topic2-Data.pdf`, `Tutorial 2.pdf`, `Tutorial 4.pdf`, guideline PDF | Uses stratification, class balance EDA, balanced class weights where supported, and recall/F2 in the discussion. | OK |
| Logistic regression | `Topic3-Supervised Learning II-V3.pdf`, guideline PDF | Uses Logistic Regression as the simple baseline model. | OK |
| Decision trees | `Topic7-Decision_Trees_Interpretability_Ensembles.pdf`, `COEN_330_Tutorial_5_Notebook.ipynb`, guideline PDF | Trains and tunes a Decision Tree model. | OK |
| Random forests / ensembles | `Topic7-Decision_Trees_Interpretability_Ensembles.pdf`, `Lecture_8_Ensemble_Learning_pw_v2.pdf`, guideline PDF | Trains and tunes Random Forest; also trains Extra Trees as another ensemble. | OK |
| Gradient boosting | `Lecture_8_Ensemble_Learning_pw_v2.pdf`, `COEN_330_Tutorial_5_Notebook.ipynb`, guideline PDF | Trains and tunes `HistGradientBoostingClassifier`, which is the final selected model. | OK |
| Baseline models | Guideline PDF, `Topic3-Supervised Learning I.pdf` general model comparison discussion | Uses Logistic Regression as the baseline before comparing tree and ensemble models. | OK |
| Hyperparameter tuning | `Topic3-Supervised Learning II-V3.pdf`, `Topic7-Decision_Trees_Interpretability_Ensembles.pdf`, guideline PDF | Tunes Decision Tree, Random Forest, and HistGradientBoostingClassifier using small validation grids, and saves all 36 validation trials in `results/hyperparameter_trials.csv`. | OK |
| Accuracy | `Topic3-Supervised Learning I.pdf`, `Topic3-Supervised Learning II-V3.pdf`, `Tutorial 1.pdf`, `Tutorial 4.pdf` | Reports accuracy for validation and test results. | OK |
| Precision | `Topic3-Supervised Learning I.pdf`, `Topic3-Supervised Learning II-V3.pdf`, guideline PDF | Reports precision and explains false alarms. | OK |
| Recall | `Topic3-Supervised Learning I.pdf`, `Topic3-Supervised Learning II-V3.pdf`, guideline PDF | Reports recall and treats missed failures as costly. | OK |
| F1-score | `Topic3-Supervised Learning II-V3.pdf`, guideline PDF | Uses F1-score as the main model selection metric. | OK |
| F2-score | Related to the course precision/recall/F-measure discussion in `Topic3-Supervised Learning II-V3.pdf` | Reports F2-score because recall is important for missed failures. | OK |
| ROC-AUC | Guideline PDF metric examples | Reports ROC-AUC for validation and test results. | OK |
| Confusion matrix | `Topic3-Supervised Learning II-V3.pdf`, `Tutorial 1.pdf`, `Tutorial 4.pdf`, guideline PDF | Saves and explains TN, FP, FN, TP counts. | OK |
| Data leakage | `Topic2-Data.pdf`, guideline PDF | Excludes `twf`, `hdf`, `pwf`, `osf`, and `rnf` from model features and keeps them only for EDA explanation. | OK |
| Reproducibility | Guideline PDF | Provides scripts, requirements, saved outputs, full tuning history, run logs, and command-line instructions. | OK |
| Model interpretation | `Topic7-Decision_Trees_Interpretability_Ensembles.pdf`, guideline PDF | Includes feature importance when available and explains model errors using the confusion matrix. | OK |
| Limitations and error analysis | Guideline PDF; evaluation discussions in `Topic3-Supervised Learning II-V3.pdf` | Discusses synthetic data, class imbalance, missed failures, false alarms, and lack of deployment readiness. | OK |
| Decision threshold tuning | `Topic3-Supervised Learning II-V3.pdf`, `Tutorial 4.pdf` | The project uses the default classifier threshold and documents threshold tuning as a future improvement. | Weak, but acceptable |
| PR-AUC / precision-recall curve | Guideline PDF lists PR-AUC for imbalanced classification | The project uses precision, recall, F1, F2, ROC-AUC, and confusion matrix, but does not report PR-AUC. This is not blocking because the requested metric set is already covered. | Weak, optional improvement |

## Summary

The project follows the main course expectations for a supervised classification project. The strongest matches are the clean train/validation/test separation, leakage prevention, preprocessing pipeline, five-model comparison, three tuned models, and confusion-matrix based error discussion.

The main optional improvements are threshold tuning and PR-AUC or precision-recall curve reporting. Those would be reasonable extensions for an imbalanced classification problem, but they are not required to defend the current project because the fixed project requirements already include recall, F2-score, ROC-AUC, and a confusion matrix.
