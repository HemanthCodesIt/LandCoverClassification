"""
Evaluation script for satellite imagery land use classification
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .model import get_model
from .utils import load_checkpoint
from .data_loader import EuroSATDataset


class ModelEvaluator:
    """
    Class for evaluating trained satellite imagery classification models
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the evaluator
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # EuroSAT class names
        self.class_names = EuroSATDataset.CLASSES
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on a dataset
        
        Args:
            data_loader: DataLoader containing the dataset
            
        Returns:
            Tuple of (predictions, ground_truth_labels)
        """
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc='Generating predictions'):
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Get predictions
                pred = output.argmax(dim=1, keepdim=False)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def evaluate_dataset(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset
        
        Args:
            data_loader: DataLoader containing the dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, labels = self.predict(data_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        f1_per_class = f1_score(labels, predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        return metrics, predictions, labels
    
    def print_evaluation_report(self, metrics: Dict, dataset_name: str = "Test"):
        """
        Print a formatted evaluation report
        
        Args:
            metrics: Dictionary containing evaluation metrics
            dataset_name: Name of the dataset being evaluated
        """
        print(f"\n{dataset_name} Set Evaluation Results")
        print("=" * 50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted Precision: {metrics['precision']:.4f}")
        print(f"Weighted Recall: {metrics['recall']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_score']:.4f}")
        
        print(f"\nPer-Class Results:")
        print("-" * 30)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:20s}: "
                  f"P={metrics['precision_per_class'][i]:.3f}, "
                  f"R={metrics['recall_per_class'][i]:.3f}, "
                  f"F1={metrics['f1_per_class'][i]:.3f}")
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            save_path: Path to save the plot
            figsize: Figure size
        """
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot raw confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1
        )
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Plot normalized confusion matrix
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax2
        )
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(
        self,
        metrics: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot per-class precision, recall, and F1-score
        
        Args:
            metrics: Dictionary containing evaluation metrics
            save_path: Path to save the plot
            figsize: Figure size
        """
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars
        bars1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision')
        bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall')
        bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score')
        
        # Customize plot
        ax.set_xlabel('Land Use Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
    
    def analyze_misclassifications(
        self,
        data_loader: DataLoader,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_samples: int = 8
    ):
        """
        Analyze and visualize misclassified samples
        
        Args:
            data_loader: DataLoader containing the dataset
            predictions: Model predictions
            labels: Ground truth labels
            num_samples: Number of misclassified samples to show
        """
        # Find misclassified indices
        misclassified_idx = np.where(predictions != labels)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return
        
        print(f"Found {len(misclassified_idx)} misclassified samples")
        
        # Randomly sample some misclassifications
        if len(misclassified_idx) > num_samples:
            sample_idx = np.random.choice(misclassified_idx, num_samples, replace=False)
        else:
            sample_idx = misclassified_idx
        
        # Get the actual images
        # Note: This is a simplified version - you might need to modify based on your data loader
        images_to_show = []
        current_idx = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            batch_size = len(data)
            batch_indices = range(current_idx, current_idx + batch_size)
            
            for i, global_idx in enumerate(batch_indices):
                if global_idx in sample_idx:
                    img = data[i]
                    images_to_show.append((img, global_idx))
                    
                if len(images_to_show) >= num_samples:
                    break
            
            current_idx += batch_size
            if len(images_to_show) >= num_samples:
                break
        
        # Plot misclassified samples
        cols = min(4, len(images_to_show))
        rows = (len(images_to_show) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (img, idx) in enumerate(images_to_show):
            row = i // cols
            col = i % cols
            
            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy
            img_np = img.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img_np)
            axes[row, col].set_title(
                f'True: {self.class_names[labels[idx]]}\n'
                f'Pred: {self.class_names[predictions[idx]]}',
                fontsize=10
            )
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(len(images_to_show), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Misclassified Samples', fontsize=14, y=1.02)
        plt.show()
    
    def get_prediction_confidence(
        self,
        data_loader: DataLoader,
        return_probabilities: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions with confidence scores
        
        Args:
            data_loader: DataLoader containing the dataset
            return_probabilities: Whether to return probability distributions
            
        Returns:
            Tuple of (predictions, confidence_scores, probabilities)
        """
        all_preds = []
        all_confidences = []
        all_probs = []
        
        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc='Computing confidence scores'):
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=1)
                
                # Get predictions and confidence scores
                confidences, predictions = torch.max(probs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                if return_probabilities:
                    all_probs.extend(probs.cpu().numpy())
        
        predictions = np.array(all_preds)
        confidences = np.array(all_confidences)
        probabilities = np.array(all_probs) if return_probabilities else None
        
        return predictions, confidences, probabilities
    
    def analyze_prediction_confidence(
        self,
        data_loader: DataLoader,
        threshold: float = 0.8
    ):
        """
        Analyze prediction confidence distribution
        
        Args:
            data_loader: DataLoader containing the dataset
            threshold: Confidence threshold for high-confidence predictions
        """
        predictions, confidences, _ = self.get_prediction_confidence(data_loader)
        
        # Confidence statistics
        print("Prediction Confidence Analysis:")
        print("-" * 40)
        print(f"Mean confidence: {confidences.mean():.4f}")
        print(f"Std confidence: {confidences.std():.4f}")
        print(f"Min confidence: {confidences.min():.4f}")
        print(f"Max confidence: {confidences.max():.4f}")
        print(f"High confidence predictions (>{threshold}): {(confidences > threshold).sum()}/{len(confidences)} ({(confidences > threshold).mean()*100:.1f}%)")
        
        # Plot confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram of confidence scores
        ax1.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(confidences.mean(), color='red', linestyle='--', label=f'Mean: {confidences.mean():.3f}')
        ax1.axvline(threshold, color='orange', linestyle='--', label=f'Threshold: {threshold}')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Confidence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence by class
        confidence_by_class = []
        for class_idx in range(len(self.class_names)):
            class_mask = predictions == class_idx
            if class_mask.sum() > 0:
                class_confidence = confidences[class_mask].mean()
                confidence_by_class.append(class_confidence)
            else:
                confidence_by_class.append(0)
        
        bars = ax2.bar(self.class_names, confidence_by_class)
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Confidence by Class')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidence_by_class):
            if conf > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_on_subset(
        self,
        data_loader: DataLoader,
        confidence_threshold: float = 0.9
    ) -> Dict[str, float]:
        """
        Evaluate model performance on high-confidence predictions only
        
        Args:
            data_loader: DataLoader containing the dataset
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary containing subset evaluation metrics
        """
        # Get predictions with confidence
        predictions, confidences, _ = self.get_prediction_confidence(data_loader)
        
        # Get ground truth labels
        all_labels = []
        for _, labels in data_loader:
            all_labels.extend(labels.numpy())
        labels = np.array(all_labels)
        
        # Filter high-confidence predictions
        high_conf_mask = confidences >= confidence_threshold
        
        if high_conf_mask.sum() == 0:
            print(f"No predictions above confidence threshold {confidence_threshold}")
            return {}
        
        filtered_preds = predictions[high_conf_mask]
        filtered_labels = labels[high_conf_mask]
        
        # Calculate metrics on filtered data
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        precision = precision_score(filtered_labels, filtered_preds, average='weighted')
        recall = recall_score(filtered_labels, filtered_preds, average='weighted')
        f1 = f1_score(filtered_labels, filtered_preds, average='weighted')
        
        metrics = {
            'subset_size': len(filtered_preds),
            'subset_percentage': len(filtered_preds) / len(predictions) * 100,
            'subset_accuracy': accuracy,
            'subset_precision': precision,
            'subset_recall': recall,
            'subset_f1_score': f1
        }
        
        print(f"\nHigh-Confidence Subset Evaluation (threshold: {confidence_threshold}):")
        print("-" * 60)
        print(f"Subset size: {metrics['subset_size']}/{len(predictions)} ({metrics['subset_percentage']:.1f}%)")
        print(f"Subset accuracy: {metrics['subset_accuracy']:.4f}")
        print(f"Subset precision: {metrics['subset_precision']:.4f}")
        print(f"Subset recall: {metrics['subset_recall']:.4f}")
        print(f"Subset F1-score: {metrics['subset_f1_score']:.4f}")
        
        return metrics
    
    def create_evaluation_summary(
        self,
        metrics: Dict,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive evaluation summary
        
        Args:
            metrics: Dictionary containing evaluation metrics
            predictions: Model predictions
            labels: Ground truth labels
            save_path: Path to save the summary
            
        Returns:
            Summary text
        """
        # Calculate additional statistics
        total_samples = len(predictions)
        correct_predictions = (predictions == labels).sum()
        
        # Per-class statistics
        cm = confusion_matrix(labels, predictions)
        
        # Create summary text
        summary = []
        summary.append("SATELLITE IMAGERY LAND USE CLASSIFICATION - EVALUATION SUMMARY")
        summary.append("=" * 70)
        summary.append("")
        
        # Overall performance
        summary.append("OVERALL PERFORMANCE:")
        summary.append("-" * 20)
        summary.append(f"Total samples evaluated: {total_samples:,}")
        summary.append(f"Correct predictions: {correct_predictions:,}")
        summary.append(f"Overall accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        summary.append(f"Weighted precision: {metrics['precision']:.4f}")
        summary.append(f"Weighted recall: {metrics['recall']:.4f}")
        summary.append(f"Weighted F1-score: {metrics['f1_score']:.4f}")
        summary.append("")
        
        # Per-class performance
        summary.append("PER-CLASS PERFORMANCE:")
        summary.append("-" * 25)
        summary.append(f"{'Class':<20} {'Samples':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
        summary.append("-" * 65)
        
        for i, class_name in enumerate(self.class_names):
            class_samples = (labels == i).sum()
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            
            summary.append(f"{class_name:<20} {class_samples:<8} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f}")
        
        summary.append("")
        
        # Best and worst performing classes
        best_class_idx = metrics['f1_per_class'].argmax()
        worst_class_idx = metrics['f1_per_class'].argmin()
        
        summary.append("CLASS PERFORMANCE HIGHLIGHTS:")
        summary.append("-" * 30)
        summary.append(f"Best performing class: {self.class_names[best_class_idx]} (F1: {metrics['f1_per_class'][best_class_idx]:.3f})")
        summary.append(f"Most challenging class: {self.class_names[worst_class_idx]} (F1: {metrics['f1_per_class'][worst_class_idx]:.3f})")
        summary.append("")
        
        # Model insights
        summary.append("MODEL INSIGHTS:")
        summary.append("-" * 15)
        
        # Calculate most confused pairs
        np.fill_diagonal(cm, 0)  # Remove diagonal for confusion analysis
        most_confused = np.unravel_index(cm.argmax(), cm.shape)
        
        summary.append(f"Most confused classes: {self.class_names[most_confused[0]]} -> {self.class_names[most_confused[1]]} ({cm[most_confused]} cases)")
        
        # Class balance analysis
        class_distribution = [(labels == i).sum() for i in range(len(self.class_names))]
        most_common_class = np.argmax(class_distribution)
        least_common_class = np.argmin(class_distribution)
        
        summary.append(f"Most common class: {self.class_names[most_common_class]} ({class_distribution[most_common_class]} samples)")
        summary.append(f"Least common class: {self.class_names[least_common_class]} ({class_distribution[least_common_class]} samples)")
        summary.append("")
        
        # Recommendations
        summary.append("RECOMMENDATIONS:")
        summary.append("-" * 15)
        
        if metrics['accuracy'] < 0.7:
            summary.append("• Model accuracy is below 70% - consider:")
            summary.append("  - Training for more epochs")
            summary.append("  - Using a larger model architecture")
            summary.append("  - Increasing data augmentation")
        elif metrics['accuracy'] < 0.9:
            summary.append("• Model performance is good but can be improved:")
            summary.append("  - Fine-tune hyperparameters")
            summary.append("  - Try ensemble methods")
            summary.append("  - Add more training data")
        else:
            summary.append("• Excellent model performance!")
            summary.append("  - Consider applying to real-world datasets")
            summary.append("  - Experiment with Indian satellite imagery")
        
        summary.append("")
        
        # Focus areas for improvement
        low_performing_classes = [i for i, f1 in enumerate(metrics['f1_per_class']) if f1 < 0.7]
        if low_performing_classes:
            summary.append("CLASSES NEEDING ATTENTION:")
            summary.append("-" * 28)
            for class_idx in low_performing_classes:
                summary.append(f"• {self.class_names[class_idx]} (F1: {metrics['f1_per_class'][class_idx]:.3f})")
                summary.append(f"  - Collect more training samples")
                summary.append(f"  - Review data quality and labeling")
                summary.append(f"  - Consider class-specific augmentation")
        
        # Join all lines
        summary_text = "\n".join(summary)
        
        # Print summary
        print(summary_text)
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(summary_text)
            print(f"\nEvaluation summary saved to {save_path}")
        
        return summary_text


def load_and_evaluate_model(
    model_path: str,
    model_name: str,
    test_loader: DataLoader,
    num_classes: int = 10,
    device: str = None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Load a trained model and evaluate it on test data
    
    Args:
        model_path: Path to the saved model checkpoint
        model_name: Name of the model architecture
        test_loader: DataLoader for test data
        num_classes: Number of output classes
        device: Device to run evaluation on
        
    Returns:
        Tuple of (metrics, predictions, labels, evaluator)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = get_model(model_name, num_classes=num_classes)
    checkpoint = load_checkpoint(model_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate
    metrics, predictions, labels = evaluator.evaluate_dataset(test_loader)
    
    # Print results
    evaluator.print_evaluation_report(metrics, "Test")
    
    # Create comprehensive summary
    evaluator.create_evaluation_summary(metrics, predictions, labels)
    
    return metrics, predictions, labels, evaluator


def compare_models(
    model_configs: List[Dict],
    test_loader: DataLoader,
    save_dir: str = './results'
) -> Dict[str, Dict]:
    """
    Compare multiple trained models
    
    Args:
        model_configs: List of dictionaries containing model configuration
                      Each dict should have 'name', 'path', 'model_name' keys
        test_loader: DataLoader for test data
        save_dir: Directory to save comparison results
        
    Returns:
        Dictionary containing results for each model
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    print("Comparing models...")
    print("=" * 60)
    
    for config in model_configs:
        name = config['name']
        model_path = config['path']
        model_name = config['model_name']
        
        print(f"\nEvaluating {name}...")
        
        try:
            metrics, preds, labels, evaluator = load_and_evaluate_model(
                model_path, model_name, test_loader
            )
            
            results[name] = {
                'metrics': metrics,
                'predictions': preds,
                'labels': labels,
                'evaluator': evaluator
            }
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    # Create comparison plots
    if len(results) > 1:
        plot_model_comparison(results, save_dir)
        create_detailed_comparison_report(results, save_dir)
    
    return results


def create_detailed_comparison_report(results: Dict, save_dir: str):
    """
    Create detailed comparison report for multiple models
    
    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save the report
    """
    report_lines = []
    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Overall comparison table
    report_lines.append("OVERALL PERFORMANCE COMPARISON:")
    report_lines.append("-" * 35)
    
    header = f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Sort models by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['accuracy'], reverse=True)
    
    for model_name, result in sorted_models:
        metrics = result['metrics']
        line = f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f}"
        report_lines.append(line)
    
    report_lines.append("")
    
    # Best model analysis
    best_model_name = sorted_models[0][0]
    best_metrics = sorted_models[0][1]['metrics']
    
    report_lines.append("BEST MODEL ANALYSIS:")
    report_lines.append("-" * 20)
    report_lines.append(f"Best performing model: {best_model_name}")
    report_lines.append(f"Accuracy: {best_metrics['accuracy']:.4f}")
    report_lines.append(f"This model performs best overall and is recommended for deployment.")
    report_lines.append("")
    
    # Per-class comparison for best classes
    report_lines.append("PER-CLASS PERFORMANCE (Best Model):")
    report_lines.append("-" * 35)
    
    class_names = EuroSATDataset.CLASSES
    for i, class_name in enumerate(class_names):
        f1_score = best_metrics['f1_per_class'][i]
        performance_level = "Excellent" if f1_score > 0.9 else "Good" if f1_score > 0.7 else "Needs Improvement"
        report_lines.append(f"{class_name:<20}: {f1_score:.3f} ({performance_level})")
    
    # Save report
    report_path = os.path.join(save_dir, 'model_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Detailed comparison report saved to {report_path}")


def plot_model_comparison(results: Dict, save_dir: str):
    """
    Plot comparison between multiple models
    
    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save plots
    """
    model_names = list(results.keys())
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Extract metrics for comparison
    comparison_data = {metric: [] for metric in metrics_to_compare}
    
    for model_name in model_names:
        metrics = results[model_name]['metrics']
        for metric in metrics_to_compare:
            comparison_data[metric].append(metrics[metric])
    
    # Create comparison plot
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics_to_compare):
        offset = (i - len(metrics_to_compare)/2 + 0.5) * width
        bars = ax.bar(x + offset, comparison_data[metric], width, 
                     label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {save_path}")
    plt.show()


def generate_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Generate a detailed classification report
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        class_names: List of class names
        save_path: Path to save the report
        
    Returns:
        Classification report as string
    """
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4
    )
    
    print("\nDetailed Classification Report:")
    print("=" * 50)
    print(report)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("Satellite Imagery Land Use Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        print(f"Classification report saved to {save_path}")
    
    return report


def batch_evaluate_models(
    models_dir: str,
    test_loader: DataLoader,
    results_dir: str = './batch_results'
) -> Dict[str, Dict]:
    """
    Evaluate all models in a directory
    
    Args:
        models_dir: Directory containing model checkpoints
        test_loader: DataLoader for test data
        results_dir: Directory to save batch evaluation results
