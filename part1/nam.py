from pathlib import Path

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
        return self.net(x)  # (batch, 1)


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare()
    model, device = fit_nam(data)
    report_metrics(model, data, device)


if __name__ == "__main__":
    main()
