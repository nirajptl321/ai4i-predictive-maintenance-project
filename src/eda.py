"""Exploratory data analysis outputs for the AI4I dataset.

This script does not train a model. It reads the processed dataset and saves
plots/tables that help explain the data in the report and presentation.
"""

from __future__ import annotations

# Matplotlib setup
import os

# Use a temporary matplotlib config folder so plot generation works in command
# line environments without writing user-specific matplotlib files.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai4i")

import matplotlib

# Agg is a non-interactive backend. It saves images to disk instead of opening
# a plot window, which is what we need for a reproducible project script.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

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

# Plot style constants
# These settings keep all generated plots consistent and readable in the report.
SAVEFIG_KWARGS = {"dpi": 180, "bbox_inches": "tight", "pad_inches": 0.2}
FAILURE_PALETTE = {0: "#4C78A8", 1: "#F58518"}


# Class balance plot
def save_class_balance_plot(df: pd.DataFrame) -> None:
    """Save a bar chart showing how many rows are failures vs non-failures."""
    # The target is imbalanced, so this plot explains why recall and F2-score
    # matter in addition to accuracy.
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4.8))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4C78A8")

    # Give the count labels room so the majority-class annotation does not hit the title.
    ax.set_ylim(0, counts.max() * 1.16)
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}\n{value / total:.1%}", ha="center", va="bottom")
    ax.set_title("Machine Failure Class Balance")
    ax.set_xlabel("Machine failure")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance.png", **SAVEFIG_KWARGS)
    plt.close(fig)


# Missing values summary
def save_missing_values_summary(df: pd.DataFrame) -> None:
    """Save missing-value counts and percentages for every column."""
    # This table is useful evidence that the processed dataset is complete.
    missing_counts = df.isna().sum()
    summary = missing_counts.rename("missing_count").to_frame()
    summary["missing_percent"] = summary["missing_count"] / len(df) * 100
    summary.to_csv(MISSING_VALUES_PATH)


# Feature distributions
def save_feature_distribution_plots(df: pd.DataFrame) -> None:
    """Save histograms/counts showing each feature by target class."""
    # Numeric features are shown as histograms. The categorical machine type is
    # shown as a count plot in the final subplot.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.flatten()

    for ax, column in zip(axes, NUMERIC_FEATURE_COLUMNS):
        sns.histplot(
            data=df,
            x=column,
            hue=TARGET_COLUMN,
            bins=35,
            element="step",
            palette=FAILURE_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("")

    type_axis = axes[-1]
    sns.countplot(
        data=df,
        x=TYPE_COLUMN,
        hue=TARGET_COLUMN,
        palette=FAILURE_PALETTE,
        legend=False,
        ax=type_axis,
    )
    type_axis.set_title("Type")
    type_axis.set_xlabel("")

    legend_handles = [
        Patch(facecolor=FAILURE_PALETTE[0], edgecolor=FAILURE_PALETTE[0], label="0"),
        Patch(facecolor=FAILURE_PALETTE[1], edgecolor=FAILURE_PALETTE[1], label="1"),
    ]
    # A shared legend keeps the small subplots from covering the distributions.
    fig.legend(handles=legend_handles, title=TARGET_COLUMN, loc="lower center", ncol=2)
    fig.suptitle("Feature Distributions by Machine Failure Target", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(PLOTS_DIR / "feature_distributions.png", **SAVEFIG_KWARGS)
    plt.close(fig)


# Correlation heatmap
def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a heatmap of numeric feature and target correlations."""
    # The heatmap gives a quick view of simple linear relationships. It is EDA
    # only; the model comparison still decides which model performs best.
    columns = NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]
    correlation_table = df[columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(
        correlation_table,
        annot=True,
        annot_kws={"size": 10},
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.85},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Model Features and Target")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", **SAVEFIG_KWARGS)
    plt.close(fig)


# Target vs feature plots
def save_target_vs_feature_plots(df: pd.DataFrame) -> None:
    """Save plots comparing each input feature against the target."""
    # Boxplots show how numeric feature ranges differ for failure vs no failure.
    # The final subplot summarizes failure rate by machine type.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.flatten()
    for ax, column in zip(axes, NUMERIC_FEATURE_COLUMNS):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column, ax=ax)
        ax.set_title(f"{column.replace('_', ' ').title()} by Target")
        ax.set_xlabel("Machine failure")
        ax.set_ylabel("")

    failure_rate = df.groupby(TYPE_COLUMN, as_index=False)[TARGET_COLUMN].mean()
    type_axis = axes[-1]
    sns.barplot(data=failure_rate, x=TYPE_COLUMN, y=TARGET_COLUMN, ax=type_axis, color="#F58518")
    type_axis.set_title("Failure Rate by Type")
    type_axis.set_xlabel("Type")
    type_axis.set_ylabel("Failure rate")

    fig.suptitle("Target vs Important Input Features", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS_DIR / "target_vs_features.png", **SAVEFIG_KWARGS)
    plt.close(fig)


# Failure mode counts
def save_failure_mode_count_plot(df: pd.DataFrame) -> None:
    """Save counts for diagnostic failure modes used only for explanation."""
    # Failure-mode columns explain what kind of failure occurred. They are not
    # model inputs because they would leak target information.
    available_modes = []
    for column in LEAKAGE_COLUMNS:
        if column in df.columns:
            available_modes.append(column)

    if available_modes:
        counts = df[available_modes].sum().sort_values(ascending=False)
    else:
        counts = pd.Series(dtype=int)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if counts.empty:
        ax.text(0.5, 0.5, "No failure mode columns found", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=counts.index.str.upper(), y=counts.values, ax=ax, color="#54A24B")
        # Extra headroom keeps bar labels clear when this image is inserted in documents.
        ax.set_ylim(0, counts.max() * 1.18)
        for index, value in enumerate(counts.values):
            ax.text(index, value, f"{int(value):,}", ha="center", va="bottom")
        ax.set_title("Failure Mode Counts (Explanation Only)")
        ax.set_xlabel("Diagnostic failure mode")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_mode_counts.png", **SAVEFIG_KWARGS)
    plt.close(fig)


# Main runner
def main() -> None:
    # The EDA script assumes preprocessing has already produced the processed
    # CSV, then writes all EDA outputs into results/.
    ensure_directories()
    processed_data = load_processed_data()

    save_class_balance_plot(processed_data)
    save_missing_values_summary(processed_data)
    save_feature_distribution_plots(processed_data)
    save_correlation_heatmap(processed_data)
    save_target_vs_feature_plots(processed_data)
    save_failure_mode_count_plot(processed_data)

    print(f"EDA plots saved to: {PLOTS_DIR}")
    print(f"Missing values summary saved to: {MISSING_VALUES_PATH}")


if __name__ == "__main__":
    main()
