# Final Submission Checklist

- [x] Copy official COEN 330 guideline PDF to `docs/COEN330_Project_Guidelines.pdf`.
- [x] Inspect `.gitignore` for GitHub readiness.
- [x] Confirm `data/raw/ai4i2020.csv` is tracked or visible to git.
- [x] Confirm `data/processed/ai4i_processed.csv` is visible to git.
- [x] Confirm `models/final_model.joblib` is visible to git.
- [x] Confirm `data/raw/ai4i2020.csv` is present.
- [x] Run `python -m src.preprocessing`.
- [x] Run `python -m src.eda`.
- [x] Run `python -m src.train`.
- [x] Run `python -m src.evaluate`.
- [x] Run `python demo/demo.py`.
- [x] Confirm `models/final_model.joblib` exists.
- [x] Confirm `results/metrics_table.csv` exists.
- [x] Confirm `results/test_metrics.csv` exists.
- [x] Confirm required plots exist in `results/plots/`.
- [x] Confirm `results/full_reproducibility_run.txt` exists.
- [x] Confirm `results/demo_output.txt` exists.
- [x] Run `git diff --check`.
- [x] Create `docs/comprehensive_project_review.md`.
- [x] Finalize team names and contribution wording.
- [x] Complete course-content audit against the official guideline and lecture/tutorial materials.
- [x] Create `docs/COURSE_CONTENT_AUDIT.md`.
- [x] Create `docs/GUIDELINE_AND_COURSE_COMPLIANCE_REVIEW.md`.
- [x] Add dataset license and citation notes for the AI4I dataset.
- [x] Regenerate `report/final_report.pdf` after the latest final report Markdown edit.
- [x] Verify `report/final_report.pdf` exists and `pdfinfo report/final_report.pdf` works.
- [ ] Visually inspect `report/final_report.pdf` before submission.
- [ ] Create final Moodle ZIP.
- [ ] Review final report for course-specific formatting requirements.
- [ ] Confirm external tools and AI-assistance disclosure follows the course policy.

PDF regeneration method used when `report/final_report.md` changes:

```bash
# Create a temporary HTML version of report/final_report.md, then convert it with LibreOffice headless.
libreoffice --headless --convert-to pdf --outdir report /tmp/ai4i_final_report.html
```
