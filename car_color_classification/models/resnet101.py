import torchvision
from torch import nn


class ResNet101(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ResNet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)

        self.model.fc = nn.Linear(2048, n_classes)

    def forward(self, X):
        logits = self.model(X)
        if self.training:
            logits = self.linear(logits)
        return logits
