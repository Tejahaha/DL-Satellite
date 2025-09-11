import torch.nn as nn
import torch.optim as optim

class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),  # 128x128 input → 32x32 after pooling
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


def get_model(num_classes: int) -> nn.Module:
    """Always return a CNN classifier."""
    return CNNClassifier(num_classes)


def get_optimizer(optimizer_name: str, model: nn.Module, lr: float = 0.001):
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    raise ValueError("Invalid optimizer name. Choose 'SGD', 'Adam', or 'RMSprop'.")
