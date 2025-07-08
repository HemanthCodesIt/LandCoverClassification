"""
Training script for satellite imagery land use classification
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import get_model
from .utils import AverageMeter, save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer class for satellite imagery classification
    
    Handles the training loop, validation, and model saving/loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './models'
    ):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on ('cuda' or 'cpu')
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        # Meters to track metrics
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)
            
            # Update meters
            loss_meter.update(loss.item(), len(data))
            acc_meter.update(accuracy, len(data))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'Acc': f'{acc_meter.avg:.4f}'
            })
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(data)
                
                # Update meters
                loss_meter.update(loss.item(), len(data))
                acc_meter.update(accuracy, len(data))
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss_meter.avg:.4f}',
                    'Acc': f'{acc_meter.avg:.4f}'
                })
        
        return loss_meter.avg, acc_meter.avg
    
    def train(self, num_epochs: int, save_every: int = 5) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_best_model()
                print(f"New best validation accuracy: {val_acc:.4f}")
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        return self.history
    
    def save_best_model(self):
        """Save the best model"""
        save_path = os.path.join(self.save_dir, 'best_model.pth')
        save_checkpoint({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, save_path)
    
    def save_checkpoint(self, epoch: int):
        """Save a training checkpoint"""
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []
        })
        
        return checkpoint.get('epoch', 0)


def create_trainer(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = None,
    save_dir: str = './models'
) -> Trainer:
    """
    Factory function to create a trainer with default settings
    
    Args:
        model_name: Name of the model architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of output classes
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on (auto-detected if None)
        save_dir: Directory to save models
        
    Returns:
        Configured Trainer instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = get_model(model_name, num_classes=num_classes)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )
    
    return trainer


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    ax3.plot(epochs, history['lr'], 'g-', label='Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)
    
    # Validation accuracy zoom
    ax4.plot(epochs, history['val_acc'], 'r-', linewidth=2)
    ax4.set_title('Validation Accuracy (Detailed)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    from .data_loader import create_data_loaders
    
    print("Setting up training example...")
    
    # Create data loaders (you would need to download the dataset first)
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir="./data",
            batch_size=32,
            download=False  # Set to True to download
        )
        
        # Create trainer
        trainer = create_trainer(
            model_name='simple_cnn',
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001
        )
        
        # Train for a few epochs
        history = trainer.train(num_epochs=5)
        
        # Plot results
        plot_training_history(history)
        
    except Exception as e:
        print(f"Could not run training example: {e}")
        print("Make sure to download the dataset first!")
        
    print("Training script test completed!")
