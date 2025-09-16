import os
import torch
from tqdm import tqdm
from CNN.EvaluateModel import evaluate


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10,
          save_best=True, scheduler_type="plateau", early_stopping_patience=5):
    best_acc = 0.0
    start_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    patience_counter = 0  # tracks epochs without improvement

    # ---------------- Load checkpoint if available ----------------
    if os.path.exists("best_model.pth"):
        print("Loading checkpoint from best_model.pth ...")
        checkpoint = torch.load("best_model.pth", map_location=device)
        try:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                best_acc = checkpoint.get("best_acc", 0.0)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"Resumed from epoch {start_epoch+1}, best_acc={best_acc:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded raw model weights only (old format). Optimizer reset.")
        except RuntimeError:
            print("⚠️ Incompatible checkpoint — starting fresh.")

    # ---------------- Setup Scheduler ----------------
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # ---------------- Training Loop ----------------
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            ncols=100,
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

            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Seen": total,
                "Acc": f"{100 * correct / total:.2f}%"
            })

        # Training metrics
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation metrics
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, silent=True)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        # Update scheduler
        if scheduler_type == "plateau":
            scheduler.step(val_accuracy)
        elif scheduler_type == "cosine":
            scheduler.step()

        # Save best checkpoint
        if save_best and val_accuracy > best_acc:
            best_acc = val_accuracy
            patience_counter = 0  # reset patience since we improved
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc
            }, "best_model.pth")
            print(f"✅ Saved new best model with val_acc={best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience counter = {patience_counter}/{early_stopping_patience}")

        # Epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Check early stopping
        if patience_counter >= early_stopping_patience:
            print("⏹️ Early stopping triggered — no improvement in validation accuracy.")
            break

    return history
