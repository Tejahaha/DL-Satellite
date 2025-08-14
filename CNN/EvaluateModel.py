import torch

def evaluate(model, val_loader, criterion, device, task_type="multiclass"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if task_type == "binary":
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy