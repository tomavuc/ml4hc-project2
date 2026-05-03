import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
HIDDEN_SIZE = 64
OUTPUT_PENALTY = 1e-3
DROPOUT = 0.1


class ExULayer(nn.Module):
    """ExU activation from the NAM paper: (x - b) * exp(w)"""

    def __init__(self, out_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_features))
        self.b = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.w, mean=4.0, std=0.5)
        nn.init.normal_(self.b, mean=0.0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.unsqueeze(1) - self.b) * torch.exp(self.w)


class FeatureNet(nn.Module):
    """Single-feature sub-network: ExU → ReLU → Dropout → Linear → scalar."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            ExULayer(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NAM(nn.Module):
    """Neural Additive Model: ŷ = β + Σᵢ fᵢ(xᵢ)."""

    def __init__(self, n_features: int, hidden: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.feature_nets = nn.ModuleList(
            [FeatureNet(hidden, dropout) for _ in range(n_features)]
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contributions = torch.cat(
            [net(x[:, i]) for i, net in enumerate(self.feature_nets)], dim=1
        )
        logit = contributions.sum(dim=1) + self.bias
        return logit, contributions


def fit_nam(data: PreparedData) -> tuple[NAM, torch.device]:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    X = torch.tensor(data.X_train, dtype=torch.float32).to(device)
    y = torch.tensor(data.y_train, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = NAM(data.X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits, contributions = model(X_batch)
            loss = criterion(logits, y_batch) + OUTPUT_PENALTY * contributions.pow(2).mean()
            loss.backward()
            optimizer.step()

    return model, device


def report_metrics(model: NAM, data: PreparedData, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(data.X_test, dtype=torch.float32).to(device))
        y_prob = torch.sigmoid(logits).cpu().numpy()

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


def plot_feature_importance(
    model: NAM, data: PreparedData, device: torch.device, out: Path
) -> None:
    model.eval()
    n_features = len(data.feature_names)

    # --- shape functions ---
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    with torch.no_grad():
        for i, name in enumerate(data.feature_names):
            x_min = float(data.X_train[:, i].min())
            x_max = float(data.X_train[:, i].max())
            x_grid = np.linspace(x_min, x_max, 200)
            x_t = torch.tensor(x_grid, dtype=torch.float32).to(device)
            fi = model.feature_nets[i](x_t).squeeze(1).cpu().numpy()

            axes[i].plot(x_grid, fi, color="#4c72b0", linewidth=1.5)
            axes[i].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[i].set_title(name)
            axes[i].set_xlabel("value (standardized)")
            axes[i].set_ylabel("fᵢ(xᵢ)")

    for ax in axes[n_features:]:
        ax.axis("off")

    fig.suptitle("NAM shape functions — learned contribution per feature", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "nam_shape_functions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with torch.no_grad():
        X_test_t = torch.tensor(data.X_test, dtype=torch.float32).to(device)
        _, contributions = model(X_test_t)
        importance = contributions.abs().mean(dim=0).cpu().numpy()

    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, max(4, n_features * 0.4)))
    ax.barh([data.feature_names[i] for i in order], importance[order], color="#4c72b0")
    ax.set_xlabel("Mean |fᵢ(xᵢ)| on test set")
    ax.set_title("NAM global feature importance")
    fig.tight_layout()
    fig.savefig(out / "nam_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save shape function and importance plots")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare()
    model, device = fit_nam(data)
    report_metrics(model, data, device)

    if args.plot:
        plot_feature_importance(model, data, device, OUTPUT_DIR)


if __name__ == "__main__":
    main()
