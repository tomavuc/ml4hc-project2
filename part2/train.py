import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from part2.data import get_class_weights, get_mean_std_lists, get_shuffled_dataset, get_train_dataset
from part2.model import *

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Train a model for Pneumonia Detection")
    parser.add_argument("--model_type", type=str, default="densenet", help="Type of model to use, densenet or resnet")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset folder")
    parser.add_argument("--output_model", type=str, default="best_model.pth", help="Path and filename to save the model weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the Adam optimizer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--use_weights", action="store_true", help="Apply class weights to the loss function to handle data imbalance")
    parser.add_argument("--shuffle", action="store_true", help="Use shuffled labels")
    
    # 2. Parse the arguments from the terminal
    args = parser.parse_args()

    if args.model_type == "resnet":
        model = resnet().to(device)
    else:
        model = densenet().to(device)

    mean_list, std_list = get_mean_std_lists(args.train_path)
    if args.shuffle:
        train_set = get_shuffled_dataset(args.train_path, mean_list, std_list)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    else:
        train_set = get_train_dataset(args.train_path, mean_list, std_list)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    weights_class = torch.tensor(get_class_weights(train_set), dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.use_weights:
        criterion = nn.CrossEntropyLoss(weight=weights_class)
    else:
        criterion = nn.CrossEntropyLoss()

    print(f"Starting training...")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")

    # Run for 5 epochs as a test
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print(f"Model saved to {args.output_model}")
    torch.save(model.state_dict(), args.output_model)