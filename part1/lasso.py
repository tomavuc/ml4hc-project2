import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

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


def main() -> None:
    data = prepare()

    model = fit_lasso(data)
    print(f"Best C = {model.C_[0]:.5f} (selected by {CV_FOLDS}-fold CV on balanced accuracy)\n")

    report_metrics(model, data)


if __name__ == "__main__":
    main()
