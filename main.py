import torch
import torch.optim as optim
import torch.nn as nn
from CNN import preprocess as pre
from CNN.EvaluateModel import evaluate
from CNN.TrainModel import train
from CNN.model import get_model , get_optimizer
from CNN.visualize import visualize_data , visualize_layerOutputs
import os
from CNN.ResultsPlot import plot_optim_results

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    results = {}

    use_cnn = True


    for opt_name in ["SGD", "Adam", "RMSprop"]:
        print(f"\n-----> Training with {opt_name} Optimizer <-----")
        model = get_model(task_type,
                          num_classes=num_classes if task_type == "multi" else None,
                          use_cnn=use_cnn,
                          ).to(device)
        optimizer = get_optimizer(opt_name, model, lr=0.001)

        train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        results[opt_name] = {"accuracy": val_acc, "loss": val_loss}
    plot_optim_results(results)

    print("\n---- Final Results ----")
    for opt_name, metrics in results.items():
        print(f"{opt_name}: Accuracy={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")

    print("visualizing cnn layers op")
    sample_img , _ = next(iter(val_loader))
    sample_img = sample_img[0].unsqueeze(0).to(device)  # keep batch dimension
    visualize_layerOutputs(model , sample_img)

if __name__ == "__main__":
    main()