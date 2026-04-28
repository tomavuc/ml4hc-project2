import os
import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader
from part2.model import densenet, resnet 
from part2.data import get_mean_std_lists, get_test_dataset, get_train_dataset, denormalize

def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = [] 

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of 'PNEUMONIA'
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate Metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    print(f"\n--- Metrics ---")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"AuROC:     {auroc:.4f}     | AuPRC:  {auprc:.4f}")

    return all_labels, all_probs

def plot_curves(labels, probs, output_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # AuROC
    ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(labels, probs):.2f})')
    ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[0].set_title('Receiver Operating Characteristic (ROC)')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend()

    # AuPRC
    ax[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {average_precision_score(labels, probs):.2f})')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend()

    plt.tight_layout()
    
    # Save the plot instead of showing it
    save_path = os.path.join(output_dir, "eval", "roc_prc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Evaluation curves saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Model")
    parser.add_argument("--model_type", type=str, required=False, default="densenet", help="Path to the saved .pth model file")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the saved .pth model file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the train dataset (for transformed samples)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Base directory for outputs")
    
    args = parser.parse_args()

    # Create directories safely (exist_ok=True prevents errors if they already exist)
    os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # (1) Load the model from arguments
    if args.model_type == "resnet":
        model = resnet().to(device)
    else:
        model = densenet().to(device)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    print(f"Loaded weights from {args.model_weights}")
    mean_list, std_list = get_mean_std_lists(args.train_path)

    test_loader = DataLoader(get_test_dataset(args.test_path, mean_list, std_list), batch_size=32, shuffle=True)
    train_loader = DataLoader(get_train_dataset(args.train_path, mean_list, std_list), batch_size=32, shuffle=True)
    
    print("Running evaluation...")
    labels, probs = evaluate_model(model, test_loader, device)
    plot_curves(labels, probs, args.output_dir)