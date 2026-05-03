import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from part1.config import OUTPUT_DIR
from part1.preprocess import PreparedData, prepare

CV_FOLDS = 5
C_GRID = np.logspace(-4, 2, 30)
RANDOM_STATE = 42


def fit_lasso(data: PreparedData) -> LogisticRegressionCV:
    model = LogisticRegressionCV(
        l1_ratios=(1,),
        solver="liblinear",
        Cs=C_GRID,
        cv=CV_FOLDS,
        scoring="balanced_accuracy",
        max_iter=1000,
        random_state=RANDOM_STATE,
        use_legacy_attributes=True,
    )
    model.fit(data.X_train, data.y_train)
    return model


def report_metrics(
    model: LogisticRegressionCV,
    data: PreparedData
) -> None:
    y_pred = model.predict(data.X_test)
    y_prob = model.predict_proba(data.X_test)[:, 1]

    bal_acc = balanced_accuracy_score(data.y_test, y_pred)
    roc_auc = roc_auc_score(data.y_test, y_prob)
    f1_macro = f1_score(data.y_test, y_pred, average="macro")

    print("=" * 72)
    print("Test-set performance")
    print("=" * 72)
    print(f"Balanced accuracy : {bal_acc:.4f}")
    print(f"F1                : {f1_macro:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print()
    print(classification_report(data.y_test, y_pred, target_names=["No disease", "Disease"]))

    n_nonzero = int(np.sum(model.coef_[0] != 0))
    print(f"Best C            : {model.C_[0]:.5f}")
    print(f"Non-zero coefs    : {n_nonzero} / {len(model.coef_[0])}")


def plot_feature_importance(
    model: LogisticRegressionCV,
    data: PreparedData,
    out: Path,
) -> None:
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    coefs_sorted = coefs[order]
    names_sorted = [data.feature_names[i] for i in order]
    colors = [
        "#dd8452" if c > 0 else "#4c72b0" if c < 0 else "#cccccc"
        for c in coefs_sorted
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, len(coefs) * 0.4)))
    ax.barh(names_sorted, coefs_sorted, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficients")
    fig.tight_layout()
    fig.savefig(out / "lasso_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}")


def fit_selected(lasso: LogisticRegressionCV, data: PreparedData, out: Path) -> None:
    mask = lasso.coef_[0] != 0
    selected_names = [n for n, m in zip(data.feature_names, mask) if m]
    X_train_sel = data.X_train[:, mask]
    X_test_sel = data.X_test[:, mask]

    print(f"Selected {mask.sum()} / {len(mask)} features: {selected_names}\n")

    model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_sel, data.y_train)

    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]
    bal_acc = balanced_accuracy_score(data.y_test, y_pred)
    roc_auc = roc_auc_score(data.y_test, y_prob)
    f1_macro = f1_score(data.y_test, y_pred, average="macro")
    print("=" * 72)
    print("Test-set performance (logistic regression on selected features)")
    print("=" * 72)
    print(f"Balanced accuracy : {bal_acc:.4f}")
    print(f"F1                : {f1_macro:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print()
    print(classification_report(data.y_test, y_pred, target_names=["No disease", "Disease"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", action="store_true",
                        help="Also run plain logistic regression on Lasso-selected features")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare()

    model = fit_lasso(data)
    print(f"Best C = {model.C_[0]:.5f} (selected by {CV_FOLDS}-fold CV on balanced accuracy)\n")

    report_metrics(model, data)
    plot_feature_importance(model, data, OUTPUT_DIR)

    if args.selected:
        fit_selected(model, data, OUTPUT_DIR)


if __name__ == "__main__":
    main()
