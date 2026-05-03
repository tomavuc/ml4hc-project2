import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from part1.config import OUTPUT_DIR
from part1.preprocess import PreparedData, prepare

RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
BACKGROUND_SAMPLES = 100


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(data: PreparedData) -> tuple[MLP, torch.device]:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    X = torch.tensor(data.X_train, dtype=torch.float32).to(device)
    y = torch.tensor(data.y_train, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = MLP(data.X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            criterion(model(X_batch).squeeze(1), y_batch).backward()
            optimizer.step()

    return model, device


def report_metrics(model: MLP, data: PreparedData, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(data.X_test, dtype=torch.float32).to(device))
        y_prob = torch.sigmoid(logits.squeeze(1)).cpu().numpy()

    y_pred = (y_prob >= 0.5).astype(int)

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


def explain_shap(model: MLP, data: PreparedData, device: torch.device, out: Path) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(data.X_test, dtype=torch.float32).to(device))
        y_pred = (torch.sigmoid(logits.squeeze(1)).cpu().numpy() >= 0.5).astype(int)

    background = torch.tensor(
        data.X_train[:BACKGROUND_SAMPLES], dtype=torch.float32
    ).to(device)
    explainer = shap.DeepExplainer(model, background)

    ev = explainer.expected_value
    base_val = float(ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
    n_features = len(data.feature_names)

    samples = {
        "positive_correct":   np.where((data.y_test == 1) & (y_pred == 1))[0],
        "positive_incorrect": np.where((data.y_test == 1) & (y_pred == 0))[0],
        "negative_correct":   np.where((data.y_test == 0) & (y_pred == 0))[0],
        "negative_incorrect": np.where((data.y_test == 0) & (y_pred == 1))[0],
    }

    for label in ("positive", "negative"):
        for i, kind in enumerate(("correct", "incorrect"), start=1):
            idx = samples[f"{label}_{kind}"][0]
            x = torch.tensor(data.X_test[[idx]], dtype=torch.float32).to(device)
            sv = explainer.shap_values(x)
            if isinstance(sv, list):
                sv = sv[0]
            sv = sv[0].squeeze(-1)

            explanation = shap.Explanation(
                values=sv,
                base_values=base_val,
                data=data.X_test[idx],
                feature_names=data.feature_names,
            )
            shap.plots.bar(explanation, max_display=n_features, show=False)
            plt.savefig(
                out / f"shap_bar_{label}_{i}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    X_test_t = torch.tensor(data.X_test, dtype=torch.float32).to(device)
    sv_all = explainer.shap_values(X_test_t)
    if isinstance(sv_all, list):
        sv_all = sv_all[0]
    sv_all = sv_all.squeeze(-1)  # (n_samples, n_features, 1) -> (n_samples, n_features)

    explanation_all = shap.Explanation(
        values=sv_all,
        base_values=np.full(len(data.y_test), base_val),
        data=data.X_test,
        feature_names=data.feature_names,
    )
    shap.plots.beeswarm(explanation_all, max_display=n_features, show=False)
    plt.savefig(out / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP plots saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap", action="store_true", help="Compute and save SHAP explanations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare()
    model, device = fit_mlp(data)
    report_metrics(model, data, device)

    if args.shap:
        explain_shap(model, data, device, OUTPUT_DIR)


if __name__ == "__main__":
    main()
