# Demo

Run from the repository root:

```bash
python demo/demo.py
```

The demo:

- Loads `models/final_model.joblib`
- Reads three normal samples and three failure samples from `data/processed/ai4i_processed.csv`
- Prints the true class, predicted class, and predicted probability of machine failure
- Prints the original processed row index and input feature values

This demo shows a few sample predictions only. Full evaluation is done by `python -m src.evaluate`. It is not deployment software.
