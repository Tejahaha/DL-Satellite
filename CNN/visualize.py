import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(train_loader, classes, num_samples=5):
    # Get a batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    images = images.numpy().transpose(0, 2, 3, 1)  # Convert CHW to HWC for matplotlib
   # images = (images * 0.5) + 0.5  # De-normalize (if normalization was applied)

# Display the batch
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        #axes[i].set_title(f"Label: {labels[i]}")
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
    plt.show()


def visualize_layerOutputs(model , input_img , layers_to_visualize=None , max_Features=16):
    model.eval()
    x = input_img.to(next(model.parameters()).device)

    if layers_to_visualize == None:
        layers_to_visualize=[0,3,6]

    with torch.no_grad():
        for idx,layers in enumerate(model.conv_layers):
            x = layers(x)
            if idx in layers_to_visualize:
                num_features = min(x.shape[1], max_Features)
                plt.figure(figsize=(12,6))
                for i in range(num_features):
                    plt.subplot(4,4,i+1)
                    plt.imshow(x[0, i].detach().cpu().numpy(), cmap='viridis')
                    plt.axis('off')
                plt.suptitle(f'layer {idx} feature maps')
                plt.show()

