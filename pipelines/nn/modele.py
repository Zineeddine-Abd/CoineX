"""
Architecture CNN VGG-like pour le pipeline NN.

  - 4 blocs convolutifs (Conv-BN-ReLU x2 + MaxPool)
  - Global Average Pooling (invariance à la translation, robuste au comptage)
  - 2 couches denses avec Dropout
  - ~1.19M paramètres pour in_channels=5

Référence cours : Semaine 12 (CNN, perceptron, descente de gradient).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        # in_channels = 5 (Lum, Sat, Sobel, Otsu, Canny)
        self.b1 = block(in_channels, 32)   # 384 -> 192
        self.b2 = block(32, 64)            # 192 -> 96
        self.b3 = block(64, 128)           # 96 -> 48
        self.b4 = block(128, 256)          # 48 -> 24

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x).flatten(1)         # (B, 256)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))            # (B, 64)
        x = self.fc2(x)                    # (B, 1)
        return x


if __name__ == "__main__":
    # Sanity check : forward pass à 384x384 avec 5 canaux
    m = CustomCNN(in_channels=5)
    x = torch.randn(2, 5, 384, 384)
    y = m(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Params : {n_params:,}")
