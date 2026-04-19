from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from part1.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    LABEL,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected heart.csv at {path}. "
            "Place the dataset there or update DATA_PATH."
        )
    return pd.read_csv(path)


def describe_schema(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Dataset schema")
    print("=" * 72)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print()
    print("Dtypes:")
    print(df.dtypes.to_string())
    print()


def describe_numeric(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Numeric features — summary statistics")
    print("=" * 72)
    cols = NUMERIC_FEATURES + BINARY_FEATURES
    print(df[cols].describe().T.to_string())
    print()


def describe_categorical(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Categorical features — value counts")
    print("=" * 72)
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts(dropna=False)
        pct = (counts / len(df) * 100).round(1)
        table = pd.DataFrame({"count": counts, "pct": pct})
        print(f"\n{col}:")
        print(table.to_string())
    print()


def describe_label(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Label distribution")
    print("=" * 72)
    counts = df[LABEL].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    table = pd.DataFrame({"count": counts, "pct": pct})
    table.index = table.index.map({0: "0 (no disease)", 1: "1 (disease)"})
    print(table.to_string())
    print(f"\nPositive rate: {df[LABEL].mean():.3f}")
    print()


def check_missing(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Missing values (NaN)")
    print("=" * 72)
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("No NaN values in any column.")
    else:
        print(missing[missing > 0].to_string())
    print()


def check_nonsensical(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Nonsensical values")
    print("=" * 72)
    findings = []

    chol_zeros = int((df["Cholesterol"] == 0).sum())
    if chol_zeros:
        findings.append(
            f"Cholesterol == 0 in {chol_zeros} rows "
            f"({chol_zeros / len(df):.1%}) — biologically impossible, "
        )

    bp_zeros = int((df["RestingBP"] == 0).sum())
    if bp_zeros:
        findings.append(
            f"RestingBP == 0 in {bp_zeros} rows — biologically impossible."
        )

    if findings:
        for line in findings:
            print(f"- {line}")
    else:
        print("No nonsensical values found.")
    print()


def plot_numeric_distributions(df: pd.DataFrame, out: Path) -> None:
    cols = NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        for label_val, sub in df.groupby(LABEL):
            ax.hist(
                sub[col],
                bins=30,
                alpha=0.55,
                label=f"{LABEL}={label_val}",
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions, split by label", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_distributions(df: pd.DataFrame, out: Path) -> None:
    cols = CATEGORICAL_FEATURES + BINARY_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        ct = (
            df.groupby([col, LABEL]).size().unstack(fill_value=0).sort_index()
        )
        ct.plot(kind="bar", stacked=False, ax=ax, width=0.75)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title=LABEL, fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Categorical feature distributions, split by label", y=1.02)
    fig.tight_layout()
    fig.savefig(
        out / "categorical_distributions.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_label_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[LABEL].value_counts().sort_index()
    ax.bar(
        [f"{LABEL}={i}" for i in counts.index],
        counts.values,
        color=["#4c72b0", "#dd8452"],
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_ylabel("count")
    ax.set_title("Label distribution")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame, out: Path) -> None:
    cols = NUMERIC_FEATURES + BINARY_FEATURES + [LABEL]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Correlation — numeric features and label")
    fig.tight_layout()
    fig.savefig(out / "numeric_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_boxplots(df: pd.DataFrame, out: Path) -> None:
    cols = NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="#4c72b0")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Boxplots — numeric features (outlier diagnostic)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "numeric_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    np.set_printoptions(suppress=True)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 40)

    df = load_data(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    describe_schema(df)
    describe_numeric(df)
    describe_categorical(df)
    describe_label(df)

    check_missing(df)
    check_nonsensical(df)

    plot_numeric_distributions(df, OUTPUT_DIR)
    plot_categorical_distributions(df, OUTPUT_DIR)
    plot_label_distribution(df, OUTPUT_DIR)
    plot_numeric_correlation(df, OUTPUT_DIR)
    plot_numeric_boxplots(df, OUTPUT_DIR)

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
