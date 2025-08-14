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