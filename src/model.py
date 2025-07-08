"""
Deep learning models for satellite imagery land use classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for satellite image classification
    
    This is a lightweight model perfect for learning and understanding
    the basics of satellite imagery classification.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Initialize the Simple CNN model
        
        Args:
            num_classes: Number of land use classes (10 for EuroSAT)
            input_channels: Number of input channels (3 for RGB)
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Input: 64x64, after 4 pooling operations: 4x4
        self.fc_input_size = 256 * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for satellite imagery
    
    Uses transfer learning with a pre-trained ResNet model,
    which is more powerful but still easy to understand.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        model_name: str = 'resnet18'
    ):
        """
        Initialize the ResNet classifier
        
        Args:
            num_classes: Number of land use classes
            pretrained: Whether to use pre-trained weights
            model_name: Which ResNet variant to use ('resnet18', 'resnet34', 'resnet50')
        """
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load the pre-trained ResNet model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Optionally freeze early layers for transfer learning
        self.freeze_backbone = False
        
    def freeze_early_layers(self, freeze: bool = True):
        """
        Freeze or unfreeze the backbone layers for transfer learning
        
        Args:
            freeze: Whether to freeze the backbone layers
        """
        self.freeze_backbone = freeze
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always allow the final layer to be trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class AttentionCNN(nn.Module):
    """
    CNN with spatial attention mechanism for satellite imagery
    
    This model incorporates attention to focus on important spatial regions,
    which can be particularly useful for satellite imagery where different
    regions of an image may have varying importance.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Initialize the Attention CNN model
        
        Args:
            num_classes: Number of land use classes
            input_channels: Number of input channels
        """
        super(AttentionCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction backbone
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.attention_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spatial attention
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Spatial attention
        attention_weights = torch.sigmoid(self.attention_conv(x))
        x = x * attention_weights
        
        # Global average pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def get_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different model architectures
    
    Args:
        model_name: Name of the model ('simple_cnn', 'resnet18', 'resnet34', 'attention_cnn')
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights (for ResNet models)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized PyTorch model
    """
    
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    
    elif model_name in ['resnet18', 'resnet34', 'resnet50']:
        return ResNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_name=model_name,
            **kwargs
        )
    
    elif model_name == 'attention_cnn':
        return AttentionCNN(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (3, 64, 64)):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (channels, height, width)
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Create a dummy input to test the model
    dummy_input = torch.randn(1, *input_size)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
    
    print("\nModel architecture:")
    print(model)


if __name__ == "__main__":
    # Test different models
    print("Testing different model architectures...\n")
    
    # Test Simple CNN
    print("=" * 50)
    simple_model = get_model('simple_cnn', num_classes=10)
    model_summary(simple_model)
    
    # Test ResNet18
    print("\n" + "=" * 50)
    resnet_model = get_model('resnet18', num_classes=10, pretrained=False)
    model_summary(resnet_model)
    
    # Test Attention CNN
    print("\n" + "=" * 50)
    attention_model = get_model('attention_cnn', num_classes=10)
    model_summary(attention_model)
    
    print("\nAll models tested successfully!")
