import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


# -----------------------
# 1. Custom CNN (your original)
# -----------------------
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


# -----------------------
# 2. Pretrained ResNet18
# -----------------------
def get_resnet18(num_classes, pretrained=True, fine_tune=True):
    # Use the new API
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze if only feature extraction
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# -----------------------
# 3. Model factory
# -----------------------
def get_model(arch, num_classes, pretrained=True, fine_tune=True):
    if arch == "cnn":
        return CNNClassifier(num_classes=num_classes)
    elif arch == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=pretrained, fine_tune=fine_tune)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# -----------------------
# 4. Optimizer getter (modernized)
# -----------------------
def get_optimizer(opt_name, model, lr=0.001, weight_decay=1e-4):
    if opt_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif opt_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
