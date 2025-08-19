import torch
import torch.nn as nn

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Training metrics
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation metrics
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, silent=True)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
