import torch
import torch.nn as nn
from torchvision import models

class SkinCancerResNet50(nn.Module):
    """
    ResNet50 para classificação binária de melanoma.
    Usa transfer learning do ImageNet com cabeça customizada (2048→256→1).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = self.base.fc.in_features
        
        self.base.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """Retorna logits (B, 1). Sigmoid aplicado na loss function."""
        return self.base(x)


