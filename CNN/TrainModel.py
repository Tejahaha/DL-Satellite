import torch
import torch.nn as nn

def train(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images,labels in train_loader:
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

        accuracy = 100 * correct / total
        print(f"Epoch[{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {accuracy:.2f}%")