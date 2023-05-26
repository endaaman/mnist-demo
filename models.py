import torch
from torch import nn



class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.layers(x)
        x = torch.softmax(x, dim=1)
        return x


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(16*4*4, 10)

    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x

if __name__ == '__main__':
    x = torch.randn(3, 1, 28, 28)
    model = ConvModel()
    y = model(x)
    print(y.shape)
    print(y)
