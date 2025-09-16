import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# ------------------------------
# 1. Show sample images with predictions
# ------------------------------
def visualize_data(loader, classes, model=None, device="cpu", num_samples=5):
    """Visualize a batch of sample images with true labels (and predictions if model is provided)."""
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images_np = images.numpy().transpose(0, 2, 3, 1)  # CHW -> HWC

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(np.clip(images_np[i], 0, 1))
        axes[i].axis('off')

        true_label = classes[labels[i].item()]
        title = f"True: {true_label}"

        if model is not None:
            model.eval()
            with torch.no_grad():
                outputs = model(images[i].unsqueeze(0).to(device))
                _, predicted = torch.max(outputs, 1)
                pred_label = classes[predicted.item()]
            title += f"\nPred: {pred_label}"

        axes[i].set_title(title, fontsize=9)

    plt.suptitle("Sample Images with Ground Truth (and Predictions)")
    plt.show()


# ------------------------------
# 2. Visualize feature maps from CNN layers
# ------------------------------
def visualize_layerOutputs(model, input_img, layers_to_visualize=None, max_features=16):
    """Visualize feature maps of selected convolutional layers."""
    model.eval()
    x = input_img.to(next(model.parameters()).device)

    # Handle custom CNN (has .conv_layers) vs ResNet
    if hasattr(model, "conv_layers"):
        layers = model.conv_layers
    else:
        # For ResNet: use its main conv + residual blocks
        layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )

    if layers_to_visualize is None:
        layers_to_visualize = [0, 3, 6]  # default indexes

    with torch.no_grad():
        for idx, layer in enumerate(layers):
            x = layer(x)
            if idx in layers_to_visualize:
                num_features = min(x.shape[1], max_features)
                plt.figure(figsize=(12, 6))
                for i in range(num_features):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(x[0, i].detach().cpu().numpy(), cmap='viridis')
                    plt.axis('off')
                plt.suptitle(f"Feature Maps - Layer {idx}")
                plt.show()

# ------------------------------
# 3. Grad-CAM style heatmap
# ------------------------------
def gradcam_heatmap(model, input_img, target_class, device="cpu"):
    """Generate a Grad-CAM style heatmap for CNN interpretability."""
    model.eval()
    input_img = input_img.to(device)

    # Hook to capture gradients
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the last conv layer
    if hasattr(model, "conv_layers"):  # your old custom CNN
        target_layer = model.conv_layers[-1]
    elif hasattr(model, "layer4"):  # ResNet / pretrained
        target_layer = model.layer4[-1]
    else:
        raise ValueError("Unsupported model architecture for Grad-CAM")
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_img)
    class_score = output[0, target_class]
    model.zero_grad()
    class_score.backward()

    # Get hooked data
    grads = gradients[0].mean(dim=[2, 3], keepdim=True)  # GAP over H,W
    act = activations[0]

    # Weighted sum
    cam = (grads * act).sum(dim=1).squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / cam.max()  # normalize 0-1

    # Resize heatmap
    cam = np.uint8(255 * cam)
    cam = np.stack([cam] * 3, axis=-1)

    forward_handle.remove()
    backward_handle.remove()

    return cam


def show_gradcam(model, input_img, classes, device="cpu"):
    """Overlay Grad-CAM heatmap on image for all class predictions."""
    model.eval()
    input_img = input_img.to(device)

    with torch.no_grad():
        outputs = model(input_img)
        _, predicted = torch.max(outputs, 1)
        target_class = predicted.item()

    cam = gradcam_heatmap(model, input_img, target_class, device=device)

    img = input_img[0].cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.4)  # overlay
    plt.title(f"Grad-CAM: Model focused on {classes[target_class]}")
    plt.axis("off")
    plt.show()


# ------------------------------
# 4. Show misclassified images
# ------------------------------
def show_misclassified(model, loader, classes, device="cpu", num_samples=5):
    """Show some misclassified examples."""
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for img, label, pred in zip(images, labels, preds):
                if label != pred:
                    misclassified.append((img.cpu(), label.item(), pred.item()))
                if len(misclassified) >= num_samples:
                    break
            if len(misclassified) >= num_samples:
                break

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, (img, true, pred) in enumerate(misclassified):
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].axis('off')
        axes[i].set_title(f"T: {classes[true]}\nP: {classes[pred]}", fontsize=9)

    plt.suptitle("Misclassified Images")
    plt.show()
