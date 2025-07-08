"""
Data loading and preprocessing for EuroSAT satellite imagery dataset
"""

import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Callable


class EuroSATDataset(data.Dataset):
    """
    Custom dataset class for EuroSAT satellite imagery data
    
    The EuroSAT dataset contains 27,000 labeled satellite images covering
    10 different land use and land cover classes from Sentinel-2 satellite.
    """
    
    # Define the 10 land use classes
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        """
        Initialize the EuroSAT dataset
        
        Args:
            root: Root directory where dataset will be stored
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            download: Whether to download the dataset
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        # Use torchvision's built-in EuroSAT dataset loader
        try:
            self.dataset = datasets.EuroSAT(
                root=root,
                transform=transform,
                target_transform=target_transform,
                download=download
            )
        except AttributeError:
            # Fallback for older PyTorch versions
            print("Using manual dataset loading (EuroSAT not available in torchvision)")
            self._setup_manual_dataset()
    
    def _setup_manual_dataset(self):
        """
        Manual setup for older PyTorch versions
        You would need to download and extract EuroSAT manually
        """
        # This is a simplified version - in practice, you'd implement
        # the full dataset loading logic here
        self.data_dir = os.path.join(self.root, "eurosat")
        if not os.path.exists(self.data_dir):
            raise RuntimeError(
                "EuroSAT dataset not found. Please download it manually or "
                "use a newer version of torchvision."
            )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label)
        """
        return self.dataset[idx]


def get_data_transforms(input_size: int = 64, augment: bool = True) -> dict:
    """
    Get data transformation pipelines for training and validation
    
    Args:
        input_size: Size to resize images to
        augment: Whether to apply data augmentation for training
        
    Returns:
        Dictionary containing 'train' and 'val' transforms
    """
    
    # Normalization values for satellite imagery (these are approximate)
    # In practice, you might want to calculate these from your specific dataset
    mean = [0.485, 0.456, 0.406]  # RGB means
    std = [0.229, 0.224, 0.225]   # RGB standard deviations
    
    if augment:
        # Training transforms with data augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Simple training transforms without augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    # Validation/test transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return {
        'train': train_transforms,
        'val': val_transforms
    }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    input_size: int = 64,
    augment: bool = True,
    download: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for EuroSAT dataset
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        train_split: Fraction of data to use for training
        input_size: Size to resize images to
        augment: Whether to apply data augmentation
        download: Whether to download the dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Get transforms
    transforms_dict = get_data_transforms(input_size=input_size, augment=augment)
    
    # Create datasets
    train_dataset = EuroSATDataset(
        root=data_dir,
        transform=transforms_dict['train'],
        download=download
    )
    
    val_dataset = EuroSATDataset(
        root=data_dir,
        transform=transforms_dict['val'],
        download=False  # Already downloaded
    )
    
    test_dataset = EuroSATDataset(
        root=data_dir,
        transform=transforms_dict['val'],
        download=False  # Already downloaded
    )
    
    # Calculate split sizes
    total_size = len(train_dataset)
    train_size = int(train_split * total_size)
    val_size = int((total_size - train_size) / 2)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create random splits
    torch.manual_seed(42)  # For reproducible splits
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size, test_size]
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def visualize_samples(data_loader: DataLoader, num_samples: int = 8):
    """
    Visualize samples from the data loader
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of samples
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        label = labels[i]
        
        # Denormalize the image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f'Class: {EuroSATDataset.CLASSES[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing EuroSAT data loader...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir="./data",
        batch_size=16,
        num_workers=2,
        download=True  # Set to True to download the dataset
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("Data loader test completed!")
