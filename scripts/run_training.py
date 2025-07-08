#!/usr/bin/env python3
"""
Main training script for satellite imagery land use classification
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import create_data_loaders
from train import create_trainer, plot_training_history
from evaluate import load_and_evaluate_model, generate_classification_report
from utils import set_seed, get_device, save_config, Timer, print_system_info


def create_config(args) -> dict:
    """
    Create configuration dictionary from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        # Data configuration
        'data': {
            'data_dir': args.data_dir,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'train_split': args.train_split,
            'input_size': args.input_size,
            'augment': args.augment
        },
        
        # Model configuration
        'model': {
            'name': args.model,
            'num_classes': 10,  # EuroSAT has 10 classes
            'pretrained': args.pretrained
        },
        
        # Training configuration
        'training': {
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'save_every': args.save_every,
            'seed': args.seed
        },
        
        # System configuration
        'system': {
            'device': args.device,
            'save_dir': args.save_dir,
            'results_dir': args.results_dir
        }
    }
    
    return config


def train_model(config: dict, verbose: bool = True):
    """
    Train the model with given configuration
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (trainer, history)
    """
    if verbose:
        print("Starting model training...")
        print("-" * 50)
    
    # Set random seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create data loaders
    if verbose:
        print("Creating data loaders...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_split=config['data']['train_split'],
        input_size=config['data']['input_size'],
        augment=config['data']['augment'],
        download=False  # Assume data is already downloaded
    )
    
    if verbose:
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    
    # Create trainer
    if verbose:
        print(f"Creating {config['model']['name']} model...")
    
    trainer = create_trainer(
        model_name=config['model']['name'],
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['model']['num_classes'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        device=config['system']['device'],
        save_dir=config['system']['save_dir']
    )
    
    if verbose:
        model_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Model parameters: {model_params:,}")
    
    # Train the model
    with Timer() as timer:
        history = trainer.train(
            num_epochs=config['training']['num_epochs'],
            save_every=config['training']['save_every']
        )
    
    if verbose:
        print(f"Training completed in {timer.elapsed():.2f} seconds")
    
    return trainer, history, test_loader


def evaluate_model(trainer, test_loader, config: dict, verbose: bool = True):
    """
    Evaluate the trained model
    
    Args:
        trainer: Trained model trainer
        test_loader: Test data loader
        config: Configuration dictionary
        verbose: Whether to print detailed information
    """
    if verbose:
        print("\nEvaluating model on test set...")
        print("-" * 50)
    
    # Load best model
    best_model_path = os.path.join(config['system']['save_dir'], 'best_model.pth')
    
    if os.path.exists(best_model_path):
        metrics, predictions, labels, evaluator = load_and_evaluate_model(
            model_path=best_model_path,
            model_name=config['model']['name'],
            test_loader=test_loader,
            num_classes=config['model']['num_classes'],
            device=config['system']['device']
        )
        
        # Save evaluation results
        results_dir = config['system']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate and save classification report
        from data_loader import EuroSATDataset
        report_path = os.path.join(results_dir, 'classification_report.txt')
        generate_classification_report(
            predictions, labels, EuroSATDataset.CLASSES, report_path
        )
        
        # Plot and save confusion matrix
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        evaluator.plot_confusion_matrix(predictions, labels, save_path=cm_path)
        
        # Plot and save per-class metrics
        metrics_path = os.path.join(results_dir, 'per_class_metrics.png')
        evaluator.plot_per_class_metrics(metrics, save_path=metrics_path)
        
        # Save metrics as JSON
        metrics_json_path = os.path.join(results_dir, 'metrics.json')
        metrics_to_save = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'per_class_precision': [float(x) for x in metrics['precision_per_class']],
            'per_class_recall': [float(x) for x in metrics['recall_per_class']],
            'per_class_f1': [float(x) for x in metrics['f1_per_class']]
        }
        
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        if verbose:
            print(f"Evaluation results saved to {results_dir}")
        
        return metrics
    
    else:
        if verbose:
            print(f"Best model not found at {best_model_path}")
        return None


def main():
    """Main function to handle command line arguments and run training"""
    parser = argparse.ArgumentParser(description="Train satellite imagery land use classification model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    parser.add_argument('--input_size', type=int, default=64,
                       help='Input image size')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Use data augmentation')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'attention_cnn'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (for ResNet models)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detected if None)')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save evaluation results')
    
    # Execution arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate existing model without training')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_device()
    
    # Create configuration
    config = create_config(args)
    
    # Print system information
    if args.verbose:
        print("Satellite Imagery Land Use Classification")
        print("=" * 50)
        print_system_info()
        print()
    
    # Save configuration
    config_path = os.path.join(config['system']['save_dir'], 'config.json')
    save_config(config, config_path)
    
    if args.evaluate_only:
        # Only evaluate existing model
        if args.verbose:
            print("Evaluation only mode - loading existing model...")
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            train_split=config['data']['train_split'],
            input_size=config['data']['input_size'],
            augment=False,  # No augmentation for evaluation
            download=False
        )
        
        # Evaluate model
        metrics = evaluate_model(None, test_loader, config, args.verbose)
        
        if metrics is None:
            print("❌ Evaluation failed - no trained model found!")
            return
    
    else:
        # Full training and evaluation
        try:
            # Train the model
            trainer, history, test_loader = train_model(config, args.verbose)
            
            # Plot training history
            results_dir = config['system']['results_dir']
            os.makedirs(results_dir, exist_ok=True)
            
            history_plot_path = os.path.join(results_dir, 'training_history.png')
            plot_training_history(history, save_path=history_plot_path)
            
            # Evaluate the model
            metrics = evaluate_model(trainer, test_loader, config, args.verbose)
            
            if args.verbose:
                print("\n🎉 Training and evaluation completed successfully!")
                print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
                if metrics:
                    print(f"Test accuracy: {metrics['accuracy']:.4f}")
                print(f"Results saved to: {results_dir}")
        
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if args.verbose:
        print("\nNext steps:")
        print("1. Check the results in the results directory")
        print("2. Experiment with different model architectures")
        print("3. Try applying the model to your own satellite imagery")
        print("4. Explore the Jupyter notebook for interactive analysis")


if __name__ == "__main__":
    main()
