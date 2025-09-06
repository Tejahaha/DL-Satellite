if isinstance(criterion, nn.BCELoss):
    labels_float = labels.float().unsqueeze(1)
    loss = criterion(outputs, labels_float)
    predicted = (outputs > 0.5).float()
    correct += (predicted.cpu() == labels.unsqueeze(1).cpu().float()).sum().item()
else:
    loss = criterion(outputs, labels)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted.cpu() == labels.cpu()).sum().item()

val_loss += loss.item()
total += labels.size(0)

accuracy = 100 * correct / total
avg_loss = val_loss / len(val_loader)

if not silent:
    print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.2f}%")
return avg_loss, accuracy


def evaluate_random_minibatch(model, val_loader, criterion, device, num_batches=1):
    """Evaluate only a few random minibatches"""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_batches = list(val_loader)
    chosen_batches = random.sample(all_batches, min(num_batches, len(all_batches)))

    with torch.no_grad():
        for images, labels in chosen_batches:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels_float = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_float)
                predicted = (outputs > 0.5).float()
                correct += (predicted.cpu() == labels.unsqueeze(1).cpu().float()).sum().item()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted.cpu() == labels.cpu()).sum().item()

            val_loss += loss.item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(chosen_batches)
    print(f"[Random MiniBatch Eval] Loss:{avg_loss:.4f}, Accuracy:{accuracy:.2f}%")
    return avg_loss, accuracy


# =============================
# Main
# =============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_to_data = "dataset"
    train_loader, val_loader, classes = pre.load_data(data_dir=path_to_data, batch_size=32)

    print(f"\nNumber of classes: {len(classes)}")
    print(f"Class names: {classes}")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print("\nDisplaying sample images...")
    pre.visualize_data(train_loader, classes)

    # Task detection
    task_type = "binary" if len(classes) == 2 else "multi"
    num_classes = len(classes)

    # Loss function
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    results = {}

    use_cnn = True

    for opt_name in ["SGD", "Adam", "RMSprop"]:
        # ---------- Mini-Batch Training ----------
        print(f"\n-----> Training with {opt_name} Optimizer (Mini-Batch) <-----")
        model = models.get_model(
            task_type,
            num_classes=num_classes if task_type == "multi" else None,
            use_cnn=use_cnn
        ).to(device)
        optimizer = models.get_optimizer(opt_name, model, lr=0.001)

        train_mini_batch(model, train_loader, val_loader, optimizer, criterion, device, epochs=5)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        results[f"{opt_name}_mini"] = {"accuracy": val_acc, "loss": val_loss}
        evaluate_random_minibatch(model, val_loader, criterion, device)

        # ---------- Full-Batch Training ----------
        print(f"\n-----> Training with {opt_name} Optimizer (Full-Batch) <-----")
        model = models.get_model(
            task_type,
            num_classes=num_classes if task_type == "multi" else None,
            use_cnn=use_cnn
        ).to(device)
        optimizer = models.get_optimizer(opt_name, model, lr=0.001)