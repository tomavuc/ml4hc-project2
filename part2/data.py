import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standards for ResNet
def get_stats(loader):
    mean = 0.0
    std = 0.0
    total_images = 0
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size
    return mean / total_images, std / total_images

def get_resize_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def get_random_erasing_transform(mean_list, std_list):
    return transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # Data Augmentation is crucial when training from scratch!
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_list, std=std_list),
    transforms.RandomErasing(
      p=0.5,
      scale=(0.3, 0.5),  # Erase between 10% and 30% of the image
      ratio=(0.3, 3.3),  # Allow for different shapes
      value=0            # 0 = Black out (good for X-rays)
    ),
])

def get_normalize_transform(mean_list, std_list):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_list, std=std_list)
    ])

def get_mean_std_lists(train_path):
    raw_train_set = datasets.ImageFolder(train_path, transform=get_resize_transform())
    calc_loader = DataLoader(raw_train_set, batch_size=64, shuffle=False, num_workers=2)
    mean, std = get_stats(calc_loader)
    return mean.tolist(), std.tolist()

def get_train_dataset(train_path, mean_list, std_list):
    return datasets.ImageFolder(train_path, transform=get_random_erasing_transform(mean_list, std_list))

def get_augmented_test_dataset(test_path, mean_list, std_list):
    return datasets.ImageFolder(test_path, transform=get_random_erasing_transform(mean_list, std_list))

def get_shuffled_dataset(train_path, mean_list, std_list):
    shuffled_set = datasets.ImageFolder(train_path, transform=get_random_erasing_transform(mean_list, std_list))
    
    labels = np.array(shuffled_set.targets)
    np.random.shuffle(labels)
    shuffled_set.targets = labels.tolist()
    new_samples = []
    for i, (path, old_label) in enumerate(shuffled_set.samples):
        new_samples.append((path, shuffled_set.targets[i]))
    
    shuffled_set.samples = new_samples
    
    return shuffled_set

def get_test_dataset(test_path, mean_list, std_list):
    return datasets.ImageFolder(test_path, transform=get_normalize_transform(mean_list, std_list))

def get_class_weights(train_set):
    # 2. Count occurrences of each class
    labels = np.array(train_set.targets)
    class_counts = np.bincount(labels) 
    total_samples = len(labels)

    print(f"Class counts: Normal={class_counts[0]}, Pneumonia={class_counts[1]}")

    return total_samples / class_counts

def denormalize(tensor, mean_list, std_list):
    """Reverses the normalization applied to the images so they can be plotted correctly."""
    clone = tensor.clone().cpu().detach()
    for t, m, s in zip(clone, mean_list, std_list):
        t.mul_(s).add_(m)
    return torch.clamp(clone, 0, 1)
