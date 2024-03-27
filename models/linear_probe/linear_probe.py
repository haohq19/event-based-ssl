import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features):
        # feature.shape = [batch_size, in_features]
        output = self.fc(features)
        return output # output.shape = [batch_size, num_classes]