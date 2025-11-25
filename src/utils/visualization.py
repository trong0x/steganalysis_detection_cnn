"""
Visualization utilities for steganalysis
File: /home/bot/steganalysis-detection_cnn/src/utils/visualization.py

Functions for plotting training curves, confusion matrices, and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# Training Visualization
# ============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training and validation curves

    Args:
        history: Dictionary with metrics history
        save_path: Path to save plot
        figsize: Figure size
    """
    metrics = ['loss', 'accuracy', 'f1']
    available_metrics = [m for m in metrics if f'train_{m}' in history or m in history]

    n_plots = len(available_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        # Training curve
        train_key = f'train_{metric}' if f'train_{metric}' in history else metric
        if train_key in history:
            ax.plot(history[train_key], label='Train', linewidth=2)

        # Validation curve
        val_key = f'val_{metric}' if f'val_{metric}' in history else f'validation_{metric}'
        if val_key in history:
            ax.plot(history[val_key], label='Validation', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_learning_rate(
    lr_history: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Plot learning rate schedule

    Args:
        lr_history: List of learning rates
        save_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(lr_history, linewidth=2, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to {save_path}")

    plt.show()


# ============================================================================
# Evaluation Visualization
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize
        save_path: Path to save plot
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Cover', 'Stego']

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path: Path to save plot
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr,
        color='#2E86AB',
        linewidth=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot Precision-Recall curve

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path: Path to save plot
        figsize: Figure size
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=figsize)
    plt.plot(
        recall, precision,
        color='#A23B72',
        linewidth=2,
        label=f'PR curve (AUC = {pr_auc:.3f})'
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")

    plt.show()


def plot_all_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None
):
    """
    Plot all evaluation metrics in a single figure

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        class_names: Names of classes
        save_dir: Directory to save plots
    """
    if class_names is None:
        class_names = ['Cover', 'Stego']

    fig = plt.figure(figsize=(18, 5))

    # Confusion Matrix
    ax1 = plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1
    )
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix', fontweight='bold')

    # ROC Curve
    ax2 = plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax3 = plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax3.plot(recall, precision, linewidth=2, label=f'AUC = {pr_auc:.3f}')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'evaluation_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation metrics saved to {save_path}")

    plt.show()


# ============================================================================
# Data Visualization
# ============================================================================

def plot_sample_images(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    n_samples: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    Plot sample images in a grid

    Args:
        images: Tensor of images (B, C, H, W)
        labels: True labels
        predictions: Predicted labels
        class_names: Names of classes
        n_samples: Number of samples to plot
        save_path: Path to save plot
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Cover', 'Stego']

    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx in range(n_samples):
        ax = axes[idx]

        # Denormalize image
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        # Plot
        ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')

        # Title
        title = ""
        if labels is not None:
            title += f"True: {class_names[labels[idx]]}"
        if predictions is not None:
            title += f"\nPred: {class_names[predictions[idx]]}"
            # Color based on correctness
            if labels is not None:
                color = 'green' if labels[idx] == predictions[idx] else 'red'
                ax.set_title(title, color=color, fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
        elif title:
            ax.set_title(title, fontsize=10)

    # Hide remaining axes
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")

    plt.show()


def plot_class_distribution(
    labels: List[int],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot class distribution

    Args:
        labels: List of labels
        class_names: Names of classes
        save_path: Path to save plot
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Cover', 'Stego']

    from collections import Counter
    counts = Counter(labels)

    plt.figure(figsize=figsize)
    bars = plt.bar(
        [class_names[i] for i in sorted(counts.keys())],
        [counts[i] for i in sorted(counts.keys())],
        color=['#2E86AB', '#A23B72']
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")

    plt.show()


# ============================================================================
# Model Interpretation
# ============================================================================

def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot feature importance

    Args:
        feature_names: Names of features
        importance_scores: Importance scores
        top_k: Number of top features to show
        save_path: Path to save plot
        figsize: Figure size
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[-top_k:]

    plt.figure(figsize=figsize)
    plt.barh(
        range(len(indices)),
        importance_scores[indices],
        color='#2E86AB'
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance saved to {save_path}")

    plt.show()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot attention heatmap

    Args:
        attention_weights: Attention weights (H, W)
        save_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)
    plt.title('Attention Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")

    plt.show()


# ============================================================================
# Comparison Visualization
# ============================================================================

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot comparison of multiple models

    Args:
        results: Dictionary with model results
        metrics: List of metrics to compare
        save_path: Path to save plot
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Prepare data
    df_data = []
    for model_name, model_metrics in results.items():
        for metric in metrics:
            if metric in model_metrics:
                df_data.append({
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Score': model_metrics[metric]
                })

    df = pd.DataFrame(df_data)

    plt.figure(figsize=figsize)
    sns.barplot(data=df, x='Metric', y='Score', hue='Model')
    plt.ylim(0, 1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")

    plt.show()


# ============================================================================
# Utility Functions
# ============================================================================

def save_all_plots(
    history: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_dir: str
):
    """
    Save all plots to directory

    Args:
        history: Training history
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        save_dir: Directory to save plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving plots to {save_path}...")

    # Training curves
    plot_training_curves(history, save_path=str(save_path / 'training_curves.png'))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=str(save_path / 'confusion_matrix.png'))

    # ROC curve
    plot_roc_curve(y_true, y_prob, save_path=str(save_path / 'roc_curve.png'))

    # PR curve
    plot_precision_recall_curve(y_true, y_prob, save_path=str(save_path / 'pr_curve.png'))

    # All metrics
    plot_all_evaluation_metrics(y_true, y_pred, y_prob, save_dir=str(save_path))

    print(f"All plots saved to {save_path}")
