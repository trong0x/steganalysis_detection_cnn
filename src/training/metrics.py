import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple


class MetricsCalculator:
    """Calculate various classification metrics"""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: 'np.ndarray | None' = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }

        # Calculate macro and weighted metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Calculate AUC if probabilities are provided
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    # For binary classification, use probability of positive class
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auc'] = 0.0

        return metrics

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: 'list[str] | None' = None
    ) -> str:
        """Generate classification report"""
        if target_names is None:
            target_names = ['Cover', 'Stego']
        report = classification_report(y_true, y_pred, target_names=target_names)
        if isinstance(report, dict):
            import json
            return json.dumps(report)
        return report

    @staticmethod
    def calculate_specificity(cm: np.ndarray) -> float:
        """Calculate specificity from confusion matrix"""
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def calculate_sensitivity(cm: np.ndarray) -> float:
        """Calculate sensitivity (recall) from confusion matrix"""
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track metrics during training"""

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, metrics: Dict[str, float]):
        """Update current metrics"""
        self.metrics = metrics

        # Add to history
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Tuple[float | None, int | None]:
        """
        Get best value of a metric and its epoch

        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min'

            return float('nan'), -1
            (best_value, best_epoch)
        """
        if metric_name not in self.history:
            return None, None

        values = self.history[metric_name]
        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:
            best_value = min(values)
            best_epoch = values.index(best_value)

        return best_value, best_epoch

    def get_history(self, metric_name: str | None = None) -> Dict:
        """Get metric history"""
        if metric_name:
            return {metric_name: self.history.get(metric_name, [])}
        return self.history

    def print_metrics(self, prefix: str = ""):
        """Print current metrics"""
        if not self.metrics:
            return

        print(f"{prefix}Metrics:")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.4f}")
