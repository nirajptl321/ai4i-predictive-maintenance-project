"""Run training, evaluation, and demo scripts."""

from __future__ import annotations

from mlp_xgboost import demo_mlp_xgboost, evaluate_mlp_xgboost, train_mlp_xgboost


def main() -> None:
    train_mlp_xgboost.main()
    evaluate_mlp_xgboost.main()
    demo_mlp_xgboost.main()


if __name__ == "__main__":
    main()
