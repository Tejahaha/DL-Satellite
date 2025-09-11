import torch
from tqdm import tqdm
from CNN.EvaluateModel import evaluate

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, save_best=True):
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # Wrap DataLoader with tqdm to show epoch progress
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,  # keep completed epoch bars visible
            ncols=100,   # set bar width
            dynamic_ncols=True
        )

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update tqdm bar with useful metrics
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Seen": total,
                "Acc": f"{100 * correct / total:.2f}%",
            })

        # Training metrics for the epoch
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation metrics
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, silent=True)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        # Save best model
        if save_best and val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

        # Epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    return history
