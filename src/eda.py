"""Exploratory data analysis outputs for the AI4I dataset."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai4i")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    LEAKAGE_COLUMNS,
    MISSING_VALUES_PATH,
    NUMERIC_FEATURE_COLUMNS,
    PLOTS_DIR,
    TARGET_COLUMN,
    TYPE_COLUMN,
)
from src.data_loading import load_processed_data
from src.utils import ensure_directories

sns.set_theme(style="whitegrid")


def save_class_balance_plot(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4C78A8")
    total = counts.sum()
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}\n{value / total:.1%}", ha="center", va="bottom")
    ax.set_title("Machine Failure Class Balance")
    ax.set_xlabel("Machine failure")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance.png", dpi=180)
    plt.close(fig)


def save_missing_values_summary(df: pd.DataFrame) -> None:
    summary = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percent=lambda table: table["missing_count"] / len(df) * 100)
    )
    summary.to_csv(MISSING_VALUES_PATH)


def save_feature_distribution_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, column in zip(axes, NUMERIC_FEATURE_COLUMNS):
        sns.histplot(data=df, x=column, hue=TARGET_COLUMN, bins=35, element="step", ax=ax)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("")

    sns.countplot(data=df, x=TYPE_COLUMN, hue=TARGET_COLUMN, ax=axes[-1])
    axes[-1].set_title("Type")
    axes[-1].set_xlabel("")
    fig.suptitle("Feature Distributions by Machine Failure Target", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    columns = NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap: Model Features and Target")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)


def save_target_vs_feature_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, column in zip(axes, NUMERIC_FEATURE_COLUMNS):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column, ax=ax)
        ax.set_title(f"{column.replace('_', ' ').title()} by Target")
        ax.set_xlabel("Machine failure")
        ax.set_ylabel("")

    failure_rate = df.groupby(TYPE_COLUMN, as_index=False)[TARGET_COLUMN].mean()
    sns.barplot(data=failure_rate, x=TYPE_COLUMN, y=TARGET_COLUMN, ax=axes[-1], color="#F58518")
    axes[-1].set_title("Failure Rate by Type")
    axes[-1].set_xlabel("Type")
    axes[-1].set_ylabel("Failure rate")
    fig.suptitle("Target vs Important Input Features", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "target_vs_features.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_failure_mode_count_plot(df: pd.DataFrame) -> None:
    available_modes = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    counts = df[available_modes].sum().sort_values(ascending=False) if available_modes else pd.Series(dtype=int)

    fig, ax = plt.subplots(figsize=(7, 4))
    if counts.empty:
        ax.text(0.5, 0.5, "No failure mode columns found", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=counts.index.str.upper(), y=counts.values, ax=ax, color="#54A24B")
        for index, value in enumerate(counts.values):
            ax.text(index, value, f"{int(value):,}", ha="center", va="bottom")
        ax.set_title("Failure Mode Counts (Explanation Only)")
        ax.set_xlabel("Diagnostic failure mode")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_mode_counts.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    df = load_processed_data()
    save_class_balance_plot(df)
    save_missing_values_summary(df)
    save_feature_distribution_plots(df)
    save_correlation_heatmap(df)
    save_target_vs_feature_plots(df)
    save_failure_mode_count_plot(df)
    print(f"EDA plots saved to: {PLOTS_DIR}")
    print(f"Missing values summary saved to: {MISSING_VALUES_PATH}")


if __name__ == "__main__":
    main()
