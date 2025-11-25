import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, cast


class SRMConv2D(nn.Module):
    """
    Spatial Rich Model (SRM) filters for steganalysis
    Pre-defined high-pass filters to detect subtle changes
    """

    def __init__(self, learnable: bool = False):
        super(SRMConv2D, self).__init__()

        # Define 30 SRM filters (3 basic + 27 advanced)
        # Basic filters: horizontal, vertical, diagonal edges
        filter_1 = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]

        filter_2 = [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, -2, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]]

        filter_3 = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, -1, 1, 0],
                    [0, 0, 1, -1, 0],
                    [0, 0, 0, 0, 0]]

        # Create 30 filters (simplified version - in practice you'd use all 30 SRM filters)
        filters = torch.FloatTensor([filter_1, filter_2, filter_3])
        filters = filters.unsqueeze(1)  # Add input channel dimension

        # Repeat for RGB channels
        filters = filters.repeat(1, 3, 1, 1)  # (3, 3, 5, 5)

        if learnable:
            self.weight = nn.Parameter(filters)
        else:
            self.register_buffer('weight', filters)

        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SRM filters"""
        return F.conv2d(x, self.weight, padding=2, groups=3)


class ConstrainedConv2D(nn.Module):
    """
    Constrained convolutional layer for steganalysis
    Forces center weight to be negative and others to be positive
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2
    ):
        super(ConstrainedConv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply constrained convolution"""
        # Apply constraints
        weight = self.weight.clone()
        center = self.kernel_size // 2

        # Make center negative
        weight[:, :, center, center] = -torch.abs(weight[:, :, center, center])

        # Make others positive (except center)
        mask = torch.ones_like(weight)
        mask[:, :, center, center] = 0
        weight = weight * mask + torch.abs(weight) * (1 - mask)

        return F.conv2d(x, weight, self.bias, self.stride, self.padding)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and apply convolution
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism (Squeeze-and-Excitation)"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Average pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)

        # Max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)

        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * attention.expand_as(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines channel and spatial attention
    """

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class StegConvBlock(nn.Module):
    """
    Specialized convolutional block for steganalysis
    Combines SRM filters with learnable features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_srm: bool = True,
        use_attention: bool = True
    ):
        super(StegConvBlock, self).__init__()

        self.use_srm = use_srm
        self.use_attention = use_attention

        # SRM preprocessing
        if use_srm:
            self.srm = SRMConv2D(learnable=False)
            conv_in_channels = in_channels + 3  # Original + SRM features
        else:
            conv_in_channels = in_channels

        # Main convolution
        self.conv1 = nn.Conv2d(conv_in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention
        if use_attention:
            self.attention = CBAM(out_channels)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SRM filters
        if self.use_srm:
            srm_features = self.srm(x)
            x = torch.cat([x, srm_features], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Attention
        if self.use_attention:
            x = self.attention(x)

        # Pooling
        x = self.pool(x)

        return x


class NoiseExtractorLayer(nn.Module):
    """
    Noise extraction layer for detecting subtle steganographic changes
    Uses high-pass filters to extract noise residual
    """

    def __init__(self):
        super(NoiseExtractorLayer, self).__init__()

        # High-pass filter kernel
        kernel = torch.FloatTensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]) / 8.0

        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel.repeat(3, 1, 1, 1)  # For RGB

        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract noise residual"""
        # Cast buffer to Tensor for typing and ensure it's on the same device/dtype as input
        kernel = cast(torch.Tensor, self.kernel).to(x.device, dtype=x.dtype)
        noise = F.conv2d(x, kernel, padding=1, groups=3)
        return noise


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion layer
    Learns to combine features from different scales
    """

    def __init__(self, num_features: int):
        super(AdaptiveFeatureFusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(num_features))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, features: list) -> torch.Tensor:
        """
        Fuse multiple features adaptively

        Args:
            features: List of feature tensors to fuse
        """
        weights = self.softmax(self.weights)

        # Ensure weights are on same device/dtype and shaped for broadcasting
        weights = weights.to(features[0].device, dtype=features[0].dtype).view(-1, 1, 1, 1)

        # Weighted sum via stacking (avoid Python int start value in sum)
        weighted = [w * f for w, f in zip(weights, features)]
        fused = torch.stack(weighted, dim=0).sum(dim=0)

        return fused


class GaussianNoise(nn.Module):
    """Add Gaussian noise during training (regularization)"""

    def __init__(self, stddev: float = 0.1):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x


class DropBlock2D(nn.Module):
    """
    DropBlock regularization for CNNs
    More effective than dropout for spatial data
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x

        # Calculate gamma
        gamma = self.drop_prob / (self.block_size ** 2)

        # Sample mask
        mask_shape = (x.shape[0], x.shape[1],
                     x.shape[2] - self.block_size + 1,
                     x.shape[3] - self.block_size + 1)

        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )

        mask = 1 - mask

        # Normalize and apply
        normalize_factor = mask.numel() / mask.sum()
        return x * mask * normalize_factor
