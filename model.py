import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class BabyCryClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # First convolutional block with larger kernel for better temporal modeling
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=40),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        
        # Stack of convolutional blocks
        self.features = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024)
        )
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input shape: [batch_size, time_steps]
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Feature extraction
        x = self.conv1(x)
        x = self.features(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
