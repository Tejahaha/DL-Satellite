import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(data_dir, batch_size=32, augment=True, val_split=0.2, seed=42):
    """
    Load dataset with stratified train/val split (same class distribution in both).
    """

    # -------------------
    # Transforms
    # -------------------
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]) if augment else transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # -------------------
    # Load whole dataset
    # -------------------
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    targets = np.array(full_dataset.targets)

    # Stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=val_split,
        random_state=seed,
        stratify=targets
    )

    # Apply different transforms for val
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=val_transforms), val_idx)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Class names
    classes = full_dataset.classes

    return train_loader, val_loader, classes
