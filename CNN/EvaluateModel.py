import torch
import torch.nn as nn
import random

def evaluate(model, val_loader, criterion, device, silent=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)

    if not silent:
        print(f"Validation Loss: {avg_loss:.4f} Validation Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy



def eval_Random_MiniBatch(model, val_loader, criterion, device, num_batch=1):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_batches = list(val_loader)
    choosenBatches = random.sample(all_batches, min(num_batch, len(all_batches)))

    with torch.no_grad():
        for images, labels in choosenBatches:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(choosenBatches)
    print(f"Validation Loss: {avg_loss:.4f} Validation Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy
