import torch
import torch.nn as nn
from CNN import preprocess as pre
from CNN.EvaluateModel import evaluate, eval_Random_MiniBatch
from CNN.TrainModel import train
from CNN.model import get_model, get_optimizer
from CNN.visualize import (
    visualize_data,
    visualize_layerOutputs,
    show_gradcam,
    show_misclassified
)
from CNN.ResultsPlot import plot_optim_results
from CNN.metrics import evaluate_metrics   # <-- NEW

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_to_data = "Satellite Data"
    train_loader, val_loader, classes = pre.load_data(data_dir=path_to_data, batch_size=32 , augment=True)

    print(f"Number of classes: {len(classes)}")
    print(f"Class names: {classes}")

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # --- Step 1: Show dataset sanity check with predictions ---
    print("\nDisplaying Sample Images from dataset (with predictions if model available)...")
    visualize_data(train_loader, classes)  # initially no model, just labels

    print(f"Using device: {device}")

    # --- Step 2: Training ---
    criterion = nn.CrossEntropyLoss()
    results = {}

    opt_name = "AdamW"
    print(f"\n-----> Training with {opt_name} Optimizer <-----")
    model = get_model("resnet18", num_classes=len(classes), pretrained=True, fine_tune=True).to(device)
    optimizer = get_optimizer(opt_name, model, lr=0.001)

    history = train(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=50,
        scheduler_type="plateau",  # or "cosine"
        early_stopping_patience=7  # stop if no improvement for 7 epochs
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    results[opt_name] = {"accuracy": val_acc, "loss": val_loss}

    plot_optim_results(results)

    print("\n---- Final Results ----")
    for opt_name, metrics in results.items():
        print(f"{opt_name}: Accuracy={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")

    # --- Step 3: Advanced Metrics ---
    print("\nComputing advanced evaluation metrics...")
    evaluate_metrics(model, val_loader, classes, device=device)

    # --- Step 4: Visualization after training ---
    print("\nVisualizing CNN internals...")

    # Take one sample from validation set
    sample_img, _ = next(iter(val_loader))
    sample_img = sample_img[0].unsqueeze(0).to(device)  # keep batch dimension

    # a) Feature maps
    visualize_layerOutputs(model, sample_img)

    # b) Grad-CAM heatmap (where CNN focused)
    show_gradcam(model, sample_img, classes, device=device)

    # c) Misclassified examples
    show_misclassified(model, val_loader, classes, device=device, num_samples=5)

    # d) Predictions on dataset samples
    visualize_data(val_loader, classes, model=model, device=device)


if __name__ == "__main__":
    main()
