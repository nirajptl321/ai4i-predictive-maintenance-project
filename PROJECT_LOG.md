# Project Log

## Setup

- Created a clean COEN 330 project structure for the AI4I 2020 Predictive Maintenance Dataset.
- Kept the raw dataset at `data/raw/ai4i2020.csv`.
- Added reproducible source scripts under `src/`.
- Copied the official guideline PDF from `~/Downloads/COEN330_Project_Guidelines_Extended_Complete_Summer2026.pdf` to `docs/COEN330_Project_Guidelines.pdf`.
- Inspected `.gitignore`; raw CSV, processed CSV, and final model are commit-visible, while `.venv/`, caches, notebooks checkpoints, temporary files, OS files, and raw ZIP downloads remain ignored.

## Preprocessing

- Loaded the raw CSV with UTF-8 BOM handling.
- Cleaned column names into snake_case.
- Dropped ID columns: `UDI` and `Product ID`.
- Preserved diagnostic failure mode columns for EDA only.
- Wrote `data/processed/ai4i_processed.csv`.

## Leakage Review

- Approved model features are defined in `src/config.py`.
- `twf`, `hdf`, `pwf`, `osf`, and `rnf` are excluded from `FEATURE_COLUMNS`.
- Modeling code uses only `FEATURE_COLUMNS`.

## EDA

- Generated class balance, feature distribution, correlation heatmap, target-vs-feature, and failure mode count plots.
- Wrote a missing values summary to `results/missing_values_summary.csv`.

## Modeling

- Used a stratified 70/15/15 train/validation/test split with `random_state=42`.
- Trained exactly five models.
- Tuned Decision Tree, Random Forest, and HistGradientBoostingClassifier with small validation grids.
- Kept Logistic Regression as the simple baseline model.
- Selected the final model by validation F1-score.

## Evaluation

- Evaluated the final saved model once on the held-out test split.
- Wrote `results/test_metrics.csv`.
- Generated confusion matrix and feature importance plots.

## Demo

- Added `demo/demo.py`.
- The demo loads the saved model and processed CSV, then prints predictions for sample rows.

## Final Packaging

- Added README, report drafts, notebooks, checklists, and reproducibility instructions.
- Full run output is captured in `results/full_reproducibility_run.txt` after the final pipeline run.
- Added `docs/comprehensive_project_review.md` to document guideline compliance evidence and final TODOs.

## GitHub Collaboration Configuration

- Verified local repository path, remote, current branch, and GitHub CLI authentication.
- Remote repository: `nirajptl321/ai4i-predictive-maintenance-project`.
- Current branch: `main`.
- Changed repository visibility to public with `gh repo edit`.
- Attempted to restrict direct pushes to `main` to only the owner username, `nirajptl321`.
- GitHub API rejected user/team push restrictions for this personal repository with: `Only organization repositories can have users and team restrictions`.
- Applied supported `main` branch protection instead: pull requests required, status checks not required, admin enforcement disabled, force pushes disabled, deletions disabled, linear history disabled, conversation resolution disabled.
- Verified branch protection with `GET /repos/nirajptl321/ai4i-predictive-maintenance-project/branches/main/protection`.
- Manual follow-up is needed only if the team wants an exact owner-only push allowlist; see `docs/github_repo_settings.md`.
