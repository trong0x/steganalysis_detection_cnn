"""
Utility helper functions for the steganalysis project
Collection of general-purpose utility functions used across the project.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import yaml
import shutil
from datetime import datetime
import hashlib


# ============================================================================
# Reproducibility & Environment Setup
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device (CPU/CUDA/MPS)

    Args:
        device: Preferred device ('cuda', 'cpu', 'mps', or None for auto)

    Returns:
        torch.device object
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    return device_obj


def check_gpu_memory():
    """Check and print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
    else:
        print("No GPU available")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")


# ============================================================================
# File & Path Operations
# ============================================================================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory

    Returns:
        Path to project root
    """
    # Assumes this file is in src/utils/
    return Path(__file__).parent.parent.parent


def find_files_with_extension(
    directory: Union[str, Path],
    extensions: List[str],
    recursive: bool = True
) -> List[Path]:
    """
    Find all files with given extensions in directory

    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.jpg', '.png'])
        recursive: Search recursively

    Returns:
        List of file paths
    """
    directory = Path(directory)
    files = []

    for ext in extensions:
        if recursive:
            files.extend(directory.rglob(f'*{ext}'))
            files.extend(directory.rglob(f'*{ext.upper()}'))
        else:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))

    return sorted(list(set(files)))


def get_file_size(file_path: Union[str, Path]) -> str:
    """
    Get human-readable file size

    Args:
        file_path: Path to file

    Returns:
        Formatted file size string
    """
    size_bytes = Path(file_path).stat().st_size

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} PB"


def copy_file_with_timestamp(
    src: Union[str, Path],
    dst_dir: Union[str, Path],
    prefix: str = ""
) -> Path:
    """
    Copy file to directory with timestamp

    Args:
        src: Source file
        dst_dir: Destination directory
        prefix: Optional prefix for filename

    Returns:
        Path to copied file
    """
    src_path = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{prefix}{timestamp}_{src_path.name}" if prefix else f"{timestamp}_{src_path.name}"
    dst_path = dst_dir / new_name

    shutil.copy2(src_path, dst_path)
    return dst_path


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module) -> str:
    """
    Get model size in MB

    Args:
        model: PyTorch model

    Returns:
        Formatted model size string
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return f"{size_mb:.2f} MB"


def print_model_summary(model: nn.Module):
    """
    Print comprehensive model summary

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {get_model_size(model)}")
    print("="*60 + "\n")


def freeze_model(model: nn.Module):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False
    print("Model frozen")


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True
    print("Model unfrozen")


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers by name

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
    print(f"Frozen layers: {layer_names}")


# ============================================================================
# Data Processing Utilities
# ============================================================================

def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize tensor to [0, 1]

    Args:
        tensor: Input tensor
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def split_dataset(
    data: List[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split dataset into train/val/test

    Args:
        data: List of data samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Shuffle before splitting
        seed: Random seed

    Returns:
        Tuple of (train, val, test) lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"

    if shuffle:
        if seed is not None:
            random.seed(seed)
        data = data.copy()
        random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def balance_classes(
    samples: List[Tuple[Any, int]],
    method: str = 'undersample'
) -> List[Tuple[Any, int]]:
    """
    Balance dataset classes

    Args:
        samples: List of (sample, label) tuples
        method: 'undersample' or 'oversample'

    Returns:
        Balanced list of samples
    """
    from collections import defaultdict

    # Group by class
    class_samples = defaultdict(list)
    for sample, label in samples:
        class_samples[label].append((sample, label))

    if method == 'undersample':
        # Undersample to minority class
        min_count = min(len(s) for s in class_samples.values())
        balanced = []
        for label, label_samples in class_samples.items():
            balanced.extend(random.sample(label_samples, min_count))

    elif method == 'oversample':
        # Oversample to majority class
        max_count = max(len(s) for s in class_samples.values())
        balanced = []
        for label, label_samples in class_samples.items():
            balanced.extend(label_samples)
            if len(label_samples) < max_count:
                additional = random.choices(
                    label_samples,
                    k=max_count - len(label_samples)
                )
                balanced.extend(additional)

    else:
        raise ValueError(f"Unknown method: {method}")

    random.shuffle(balanced)
    return balanced


# ============================================================================
# Time & Progress Utilities
# ============================================================================

def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def estimate_remaining_time(
    current_step: int,
    total_steps: int,
    elapsed_time: float
) -> str:
    """
    Estimate remaining time

    Args:
        current_step: Current step number
        total_steps: Total number of steps
        elapsed_time: Time elapsed so far

    Returns:
        Formatted remaining time string
    """
    if current_step == 0:
        return "Unknown"

    time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    remaining_time = time_per_step * remaining_steps

    return format_time(remaining_time)


# ============================================================================
# Serialization Utilities
# ============================================================================

def save_json(data: Dict, path: Union[str, Path], indent: int = 2):
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Dict:
    """
    Load JSON file

    Args:
        path: JSON file path

    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict, path: Union[str, Path]):
    """
    Save dictionary to YAML file

    Args:
        data: Dictionary to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_yaml(path: Union[str, Path]) -> Dict:
    """
    Load YAML file

    Args:
        path: YAML file path

    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# Checksum & Validation
# ============================================================================

def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Get file hash

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)

    Returns:
        Hex hash string
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def verify_file_integrity(
    file_path: Union[str, Path],
    expected_hash: str,
    algorithm: str = 'md5'
) -> bool:
    """
    Verify file integrity using hash

    Args:
        file_path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm

    Returns:
        True if hash matches
    """
    actual_hash = get_file_hash(file_path, algorithm)
    return actual_hash == expected_hash


# ============================================================================
# Pretty Printing
# ============================================================================

def print_dict(data: Dict, indent: int = 0, max_depth: int = 5):
    """
    Pretty print nested dictionary

    Args:
        data: Dictionary to print
        indent: Current indentation level
        max_depth: Maximum nesting depth
    """
    if indent >= max_depth:
        print("  " * indent + "...")
        return

    for key, value in data.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1, max_depth)
        elif isinstance(value, list) and len(value) > 10:
            print("  " * indent + f"{key}: [List with {len(value)} items]")
        else:
            print("  " * indent + f"{key}: {value}")


def print_separator(char: str = "=", length: int = 60):
    """Print a separator line"""
    print(char * length)


def print_header(text: str, char: str = "=", length: int = 60):
    """Print a formatted header"""
    print_separator(char, length)
    print(text.center(length))
    print_separator(char, length)


# ============================================================================
# Experiment Management
# ============================================================================

def create_experiment_dir(
    base_dir: Union[str, Path],
    experiment_name: str
) -> Path:
    """
    Create timestamped experiment directory

    Args:
        base_dir: Base experiments directory
        experiment_name: Name of experiment

    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def save_experiment_info(
    exp_dir: Union[str, Path],
    config: Dict,
    model_summary: Optional[str] = None
):
    """
    Save experiment information

    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary
        model_summary: Optional model summary text
    """
    exp_dir = Path(exp_dir)

    # Save config
    save_json(config, exp_dir / "config.json")
    # Save info file
    info = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": config.get("experiment_name", "unknown"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": torch.__version__,
    }

    save_json(info, exp_dir / "experiment_info.json")

    # Save model summary if provided
    if model_summary:
        with open(exp_dir / "model_summary.txt", 'w') as f:
            f.write(model_summary)


# ============================================================================
# Quick Tests
# ============================================================================

if __name__ == "__main__":
    print("Testing helpers.py utilities...")

    # Test seed setting
    set_seed(42)

    # Test device detection
    device = get_device()

    # Test file operations
    test_dir = ensure_dir("./test_output")
    print(f"Created directory: {test_dir}")

    # Test model utilities
    model = nn.Linear(10, 5)
    print_model_summary(model)

    # Test time formatting
    print(f"Time: {format_time(3665.5)}")

    print("\nAll tests passed!")
