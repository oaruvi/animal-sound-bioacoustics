"""
Bioacoustic Backbone Models Architecture Definition
CSCI 2026 - Passive Acoustic Monitoring Benchmark

This module implements and adapts the three candidate neural backbones evaluated across
the 18 experimental configurations:
1. EfficientNet-B1 (ImageNet pre-trained vision backbone - Winner)
2. ResNet50 V2 (Residual vision backbone)
3. PANNs CNN14 (Pre-trained Audio Neural Network backbone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class PANNsCNN14(nn.Module):
    """
    Pre-trained Audio Neural Network (CNN14) adapted for 206 Neotropical species classification.
    Ref: Kong et al., 2020 (PANNs: Large-Scale Pretrained Audio Neural Networks).
    """
    def __init__(self, num_classes: int = 206, in_channels: int = 3, dropout_rate: float = 0.2):
        super(PANNsCNN14, self).__init__()
        
        self.conv_block1 = self._make_block(in_channels, 64)
        self.conv_block2 = self._make_block(64, 128)
        self.conv_block3 = self._make_block(128, 256)
        self.conv_block4 = self._make_block(256, 512)
        self.conv_block5 = self._make_block(512, 1024)
        self.conv_block6 = self._make_block(1024, 2048)
        
        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_audioset = nn.Linear(2048, num_classes)

    def _make_block(self, in_c: int, out_c: int):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 3, 224, 224]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        
        # Global Average & Max Pooling
        x = torch.mean(x, dim=3) + torch.max(x, dim=3)[0]
        x = torch.mean(x, dim=2)
        
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc_audioset(x)
        return logits


def get_model(
    model_name: str = "efficientnet_b1",
    num_classes: int = 206,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> nn.Module:
    """
    Factory function to instantiate adapted backbone architectures.

    Args:
        model_name: One of ['efficientnet_b1', 'resnet50', 'panns_cnn14']
        num_classes: Target species count (default: 206)
        pretrained: Whether to load ImageNet/AudioSet weights
        dropout_rate: Dropout probability in the classification head

    Returns:
        nn.Module: PyTorch model ready for training/inference
    """
    model_name = model_name.lower()

    if model_name == "efficientnet_b1":
        if HAS_TIMM:
            model = timm.create_model(
                "efficientnet_b1",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout_rate
            )
        else:
            from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            model = efficientnet_b1(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes)
            )

    elif model_name == "resnet50":
        if HAS_TIMM:
            model = timm.create_model(
                "resnet50d",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout_rate
            )
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )

    elif model_name in ["panns_cnn14", "panns"]:
        model = PANNsCNN14(num_classes=num_classes, dropout_rate=dropout_rate)

    else:
        raise ValueError(f"Unsupported model_name: '{model_name}'. Choose from ['efficientnet_b1', 'resnet50', 'panns_cnn14']")

    return model


if __name__ == "__main__":
    # Sanity check forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    for arch in ["efficientnet_b1", "resnet50", "panns_cnn14"]:
        net = get_model(arch, num_classes=206, pretrained=False)
        output = net(dummy_input)
        print(f"Architecture: {arch:<15} | Input: {tuple(dummy_input.shape)} | Output: {tuple(output.shape)}")
