import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy with class weights"""

    def __init__(self, weight: Optional[torch.Tensor] = None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.ce_loss(inputs, targets)


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing"""

    def __init__(
        self,
        num_classes: int = 2,
        smoothing: float = 0.1
    ):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class CombinedLoss(nn.Module):
    """Combination of multiple losses"""

    def __init__(
        self,
        ce_weight: float = 0.7,
        focal_weight: float = 0.3,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.ce_weight * ce + self.focal_weight * focal


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments

    Returns:
        Loss function module
    """
    losses = {
        'ce': nn.CrossEntropyLoss,
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
        'weighted_ce': WeightedCrossEntropyLoss,
        'label_smoothing': LabelSmoothingLoss,
        'combined': CombinedLoss,
    }

    if loss_name not in losses:
        raise ValueError(
            f"Unknown loss: {loss_name}. "
            f"Available: {list(losses.keys())}"
        )

    return losses[loss_name](**kwargs)
