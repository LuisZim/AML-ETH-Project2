import torch
import torch.nn as nn
import torchvision.models as models

class BBoxRegressor(nn.Module):
    """
    Bounding Box Regression Model.
    
    Input: Image (B, 3, H, W)
    Output: Normalized bbox (B, 4) in format [x_min, y_min, x_max, y_max] ∈ [0, 1]
    """
    def __init__(self, backbone='resnet18', pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.encoder = models.resnet18(weights='DEFAULT' if pretrained else None)
            in_features = 512
        elif backbone == 'mobilenet_v3_small':
            self.encoder = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
            in_features = 576
        elif backbone == 'efficientnet_b0':
            self.encoder = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            in_features = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        if backbone == 'resnet18':
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Remove fc layer
        elif backbone == 'mobilenet_v3_small':
            self.encoder = self.encoder.features  # Use only features
        elif backbone == 'efficientnet_b0':
            self.encoder = self.encoder.features  # Use only features
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        # Backbone
        features = self.encoder(x)  # (B, C, H, W)
        
        # Global Average Pooling
        pooled = self.gap(features)  # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        
        # Regression Head
        bbox = self.head(pooled)  # (B, 4) in [0, 1]
        
        return bbox