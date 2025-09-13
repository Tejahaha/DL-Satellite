import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNClassifier, self).__init__()

        # Convolutional layers (with batchnorms)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # conv 0
            nn.BatchNorm2d(16),                                    # conv 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # conv 3

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # conv 4
            nn.BatchNorm2d(32),                                    # conv 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # conv 7

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # conv 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),  # fc 0
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),           # fc 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),           # fc 2
            nn.ReLU(),
            nn.Linear(128, 64),            # fc 3
            nn.ReLU(),
            nn.Linear(64, num_classes)     # fc 4
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x


# Factory method
def get_model(num_classes=4):
    return CNNClassifier(num_classes=num_classes)


# Optimizer getter
def get_optimizer(opt_name, model, lr=0.001):
    if opt_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
