import torch
import torch.optim as optim
import torch.nn as nn
from CNN import preprocess as pre
from CNN.EvaluateModel import evaluate
from CNN.TrainModel import train
from CNN.model import get_model
from CNN.visualize import visualize_data



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_to_data = "Satellite Data"
    train_loader, val_loader, classes = pre.load_data(data_dir=path_to_data, batch_size=32)

    print(f"Number of classes: {len(classes)}")
    print(f"Class names: {classes}")

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    print("\nDisplaying Sample Images from dataset....")
    visualize_data(train_loader, classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    task_type = "binary" if len(classes) == 2 else "multi"
    num_classes = len(classes) if task_type == "multi" else None
    model = get_model(task_type, num_classes=num_classes).to(device)

    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining Model...")
    train(model, train_loader, optimizer, criterion, device, epochs=10)

    print(f"\nEvaluating on Validation set...")
    evaluate(model, val_loader, criterion, device)


if __name__ == "__main__":
    main()
