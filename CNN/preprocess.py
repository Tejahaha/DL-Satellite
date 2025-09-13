import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_data(data_dir, batch_size=32, augment=False):

    # ---------- Base Transform ----------
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # resize all images to 128x128
        transforms.ToTensor(),
    ])

    # ---------- Augmentation Transform ----------
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        train_transform = base_transform

    # ---------- Load Dataset ----------
    # Note: torchvision expects structure like:
    # data_dir/
    #    cloudy/
    #    desert/
    #    green_area/
    #  water/
    dataset = datasets.ImageFolder(root=data_dir, transform=base_transform)
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # ---------- Split into Train / Val ----------
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # ---------- Data Loaders ----------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes
