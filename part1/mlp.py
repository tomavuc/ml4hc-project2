import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from part1.config import OUTPUT_DIR
from part1.preprocess import PreparedData, prepare

RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3


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
        return self.net(x).squeeze(1)


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
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

    return model, device


def report_metrics(model: MLP, data: PreparedData, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(data.X_test, dtype=torch.float32).to(device))
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
    model, device = fit_mlp(data)
    report_metrics(model, data, device)


if __name__ == "__main__":
    main()
