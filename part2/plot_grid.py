import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from part2.model import densenet, resnet 
from part2.data import get_mean_std_lists, get_test_dataset, get_train_dataset, denormalize
from part2.gradcam import GradCAM
from part2.ig import integrated_gradients

def plot_basic_grid_horizontal(loader, mean_list, std_list, output_path, suptitle, n_samples=5):
    """Plots a 2x5 grid: Top row is Normal, Bottom row is Pneumonia."""
    normal_samples = []
    pneumonia_samples = []

    for images, labels in loader:
        for i in range(len(images)):
            label = labels[i].item()
            if label == 0 and len(normal_samples) < n_samples:
                normal_samples.append(images[i])
            elif label == 1 and len(pneumonia_samples) < n_samples:
                pneumonia_samples.append(images[i])
            
            if len(normal_samples) == n_samples and len(pneumonia_samples) == n_samples:
                break
        if len(normal_samples) == n_samples and len(pneumonia_samples) == n_samples:
            break

    # 2 rows, 5 columns
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    fig.suptitle(suptitle, fontsize=16, y=1.02, fontweight='bold')
    
    for i in range(n_samples):
        # Row 0: Normal
        img_norm = denormalize(normal_samples[i], mean_list, std_list).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_norm)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Healthy (NORMAL)", fontsize=12, pad=10, loc='left')

        # Row 1: Pneumonia
        img_pneu = denormalize(pneumonia_samples[i], mean_list, std_list).permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_pneu)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Disease (PNEUMONIA)", fontsize=12, pad=10, loc='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_xai_horizontal(model, gcam_obj, xai_type, loader, mean_list, std_list, device, output_path, suptitle, n_samples=5, category="NORMAL"):
    """Plots a 2x5 grid for a SINGLE algorithm: Top row is Original, Bottom row is XAI."""
    label_idx = 0 if category == "NORMAL" else 1
    samples_found = 0

    # 2 rows, 5 columns
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    fig.suptitle(suptitle, fontsize=16, y=1.02, fontweight='bold')
    
    cmap_gcam = plt.get_cmap('jet')
    cmap_ig = plt.get_cmap('hot')

    for images, labels in loader:
        for i in range(len(images)):
            if labels[i] == label_idx and samples_found < n_samples:
                img_tensor = images[i].unsqueeze(0).to(device)
                img_plot = denormalize(img_tensor[0], mean_list, std_list).permute(1, 2, 0).numpy()
                
                # Row 0: Original X-Ray
                axes[0, samples_found].imshow(img_plot)
                axes[0, samples_found].axis('off')
                if samples_found == 0: 
                    axes[0, samples_found].set_title("Original X-Ray", fontsize=12, pad=10, loc='left')

                # Row 1: The Requested XAI Heatmap
                if xai_type == "gcam":
                    mask = gcam_obj.generate(img_tensor, class_idx=label_idx)
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    heatmap_tensor = F.interpolate(mask_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                    heatmap_gcam = heatmap_tensor.squeeze().numpy()
                    heatmap_rgb = cmap_gcam(heatmap_gcam)[:, :, :3]
                    overlay = np.clip(heatmap_rgb * 0.4 + img_plot * 0.6, 0, 1)
                    
                    axes[1, samples_found].imshow(overlay)
                    if samples_found == 0: 
                        axes[1, samples_found].set_title("Grad-CAM Overlay", fontsize=12, pad=10, loc='left')

                elif xai_type == "ig":
                    attr = integrated_gradients(img_tensor, model, label_idx, device, steps=100)
                    heatmap_ig = np.sum(np.abs(attr), axis=0)
                    vmax = np.percentile(heatmap_ig, 99)
                    if vmax > 0: 
                        heatmap_ig = np.clip(heatmap_ig / vmax, 0, 1)
                    ig_rgb = cmap_ig(heatmap_ig)[:, :, :3]
                    
                    axes[1, samples_found].imshow(ig_rgb)
                    if samples_found == 0: 
                        axes[1, samples_found].set_title("Integrated Gradients", fontsize=12, pad=10, loc='left')

                axes[1, samples_found].axis('off')
                samples_found += 1
                
            if samples_found >= n_samples: break
        if samples_found >= n_samples: break
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate horizontal 2x5 grids dynamically.")
    parser.add_argument("--model_type", type=str, default="densenet", help="Model architecture")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the saved .pth model file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the train dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/final_grids", help="Directory to save grids")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    print("Loading model and weights...")
    if args.model_type == "resnet":
        model = resnet().to(device)
        target_layer = model.layer4
    else:
        model = densenet().to(device)
        target_layer = model.denseblock4

    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval() 
    gcam = GradCAM(model, target_layer)

    # 2. Load Data
    print("Calculating stats and preparing loaders...")
    mean_list, std_list = get_mean_std_lists(args.train_path)
    
    test_loader = DataLoader(get_test_dataset(args.test_path, mean_list, std_list), batch_size=32, shuffle=False)
    train_loader = DataLoader(get_train_dataset(args.train_path, mean_list, std_list), batch_size=32, shuffle=True)

    # 3. Generate Basic Grids
    print("\n--- Generating Basic Grids ---")
    plot_basic_grid_horizontal(
        test_loader, mean_list, std_list, 
        output_path=os.path.join(args.output_dir, "01_original_samples_horizontal.png"),
        suptitle="Original Test Dataset Samples"
    )

    plot_basic_grid_horizontal(
        train_loader, mean_list, std_list, 
        output_path=os.path.join(args.output_dir, "02_augmented_samples_horizontal.png"),
        suptitle="Data Augmentation Pipeline"
    )

    # 4. Generate Grad-CAM Grids
    print("\n--- Generating Grad-CAM Grids ---")
    plot_xai_horizontal(
        model, gcam, "gcam", test_loader, mean_list, std_list, device,
        output_path=os.path.join(args.output_dir, "03_gcam_normal.png"),
        suptitle="Grad-CAM: NORMAL", category="NORMAL"
    )
    plot_xai_horizontal(
        model, gcam, "gcam", test_loader, mean_list, std_list, device,
        output_path=os.path.join(args.output_dir, "04_gcam_pneumonia.png"),
        suptitle="Grad-CAM: PNEUMONIA", category="PNEUMONIA"
    )

    # 5. Generate Integrated Gradients Grids
    print("\n--- Generating Integrated Gradients Grids ---")
    plot_xai_horizontal(
        model, gcam, "ig", test_loader, mean_list, std_list, device,
        output_path=os.path.join(args.output_dir, "05_ig_normal.png"),
        suptitle="Integrated Gradients: NORMAL", category="NORMAL"
    )
    plot_xai_horizontal(
        model, gcam, "ig", test_loader, mean_list, std_list, device,
        output_path=os.path.join(args.output_dir, "06_ig_pneumonia.png"),
        suptitle="Integrated Gradients: PNEUMONIA", category="PNEUMONIA"
    )

    print(f"\nSuccess! All horizontal grids saved dynamically to {args.output_dir}")