import torch
from torchvision import transforms , datasets
from torch.utils.data import DataLoader
#image prosessing
from matplotlib import pyplot as plt
import numpy as np

def load_data(data_dir , batch_size=32 , augment=False):
    #first we have to convert the dataset into tensor since torch cant use arrays
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()])

    if augment:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5 , 0.5 , 0.5], std=[0.5 , 0.5 , 0.5])
        ])

    #here dataset is converted to tensor dataset
    dataset = datasets.ImageFolder(root= f"{data_dir}", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset , val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader , dataset.classes