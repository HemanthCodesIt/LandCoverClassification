"""
Utility functions for satellite imagery land use classification project
"""

import os
import torch
import numpy as np
import random
import time
from typing import Dict, Any, Optional
import json
import yaml


class AverageMeter:
    """
    Computes and stores the average and current value
    Useful for tracking metrics during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value
        
        Args:
            val: New value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state and other info
        filepath: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'model_best.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    filepath: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to the checkpoint file
        device: Device to load the checkpoint on
        
    Returns:
        Dictionary containing the checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """
    Get the best available device for training
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    return device


def create_directory(path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the config file
    """
    create_directory(os.path.dirname(filepath))
    
    # Determine file format based on extension
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError("Config file must be .json or .yaml/.yml")
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        filepath: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found at {filepath}")
    
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            config = json.load(f)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Config file must be .json or .yaml/.yml")
    
    print(f"Configuration loaded from {filepath}")
    return config


class Timer:
    """
    Simple timer utility for measuring execution time
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        return elapsed
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        return time.time() - self.start_time
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit"""
        elapsed = self.stop()
        print(f"Elapsed time: {elapsed:.2f} seconds")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def calculate_dataset_stats(data_loader, channels: int = 3) -> Dict[str, torch.Tensor]:
    """
    Calculate mean and standard deviation of a dataset
    
    Args:
        data_loader: DataLoader for the dataset
        channels: Number of channels in the images
        
    Returns:
        Dictionary containing mean and std tensors
    """
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    total_samples = 0
    
    print("Calculating dataset statistics...")
    
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    
    return {'mean': mean, 'std': std}


def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print stopping information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
            
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
        
        return self.early_stop


def print_system_info():
    """Print system information for debugging"""
    print("System Information:")
    print("-" * 30)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print(f"Number of CPU cores: {torch.get_num_threads()}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test timer
    print("\nTesting Timer:")
    with Timer() as timer:
        time.sleep(0.1)  # Simulate some work
    
    # Test seed setting
    print("\nTesting seed setting:")
    set_seed(42)
    print("Random seed set to 42")
    
    # Test system info
    print("\n")
    print_system_info()
    
    # Test early stopping
    print("\nTesting Early Stopping:")
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    # Simulate some losses
    losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, loss in enumerate(losses):
        stop = early_stopping(loss, None)  # No model for testing
        print(f"Epoch {i+1}, Loss: {loss}, Stop: {stop}")
        if stop:
            break
    
    print("\nUtility functions test completed!")
