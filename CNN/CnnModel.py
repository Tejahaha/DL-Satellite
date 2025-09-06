import torch
import torch.nn as nn


class CNNClassifier(nn.module):
    def __init__(self , num_classes):
        nn.Conv2