import torch
import torch.nn as nn
import torch.optim as optim


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class MultiClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            # No Sigmoid/Softmax here, CrossEntropyLoss handles it
        )

    def forward(self, x):
        return self.network(x)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),  # for 128x128 input → 32x32 after pooling
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


# def get_model(task_type, num_classes=None):
#     if task_type == "binary":
#         return BinaryClassifier()
#     elif task_type == "multi":
#         if num_classes is None:
#             raise ValueError("num_classes must be provided for multi-class classification")
#         return MultiClassifier(num_classes)
#     else:
#         raise ValueError("Invalid task_type. Choose 'binary' or 'multi'.")

def get_model(task_type, num_classes=None, use_cnn=False):
    """Factory to select model type"""
    if use_cnn:
        if num_classes is None:
            raise ValueError("num_classes must be specified when using CNN")
        return CNNClassifier(num_classes)
    elif task_type == "binary":
        return BinaryClassifier()
    elif task_type == "multi":
        if num_classes is None:
            raise ValueError("num_classes must be specified for multi-class classification")
        return MultiClassifier(num_classes)
    else:
        raise ValueError("Invalid task type, choose 'binary' or 'multi'")

def get_optimizer(optimizer_name,model,lr=0.001):
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr,momentum=0.9)
    elif optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invlid optimizer name ")