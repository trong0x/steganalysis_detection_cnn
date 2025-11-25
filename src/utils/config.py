"""
Configuration management utilities
File: /home/bot/steganalysis-detection_cnn/src/utils/config.py

Centralized configuration system for the steganalysis project.
Supports YAML and JSON formats with validation and defaults.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import os


@dataclass
class DataConfig:
    """Data configuration for training and validation"""

    # Directories
    cover_train_dir: str = "data/raw/images/cover"
    stego_train_dir: str = "data/raw/images/stego"
    cover_val_dir: str = ""
    stego_val_dir: str = ""
    cover_test_dir: str = ""
    stego_test_dir: str = ""

    # Image settings
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: str = "medium"  # 'light', 'medium', 'strong', 'auto'

    # Data split
    val_split: float = 0.2
    test_split: float = 0.1
    shuffle_data: bool = True

    # Class balance
    use_balanced_sampling: bool = False

    def validate(self):
        """Validate configuration"""
        if self.img_size <= 0:
           raise ValueError("img_size must be positive")
        if self.batch_size <= 0:
           raise ValueError("batch_size must be positive")
        if not (0 <= self.val_split < 1):
           raise ValueError("val_split must be in [0, 1)")
        if not (0 <= self.test_split < 1):
           raise ValueError("test_split must be in [0, 1)")
        if self.augmentation_strength not in ['light', 'medium', 'strong', 'auto', 'none']:
           raise ValueError(f"augmentation_strength must be one of: light, medium, strong, auto, none. Got: {self.augmentation_strength}")


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    # Model selection
    model_name: str = "resnet50_steg"
    num_classes: int = 2

    # Transfer learning
    pretrained: bool = True
    freeze_backbone: bool = False
    freeze_epochs: int = 0  # Unfreeze after N epochs

    # Architecture
    dropout: float = 0.5
    use_attention: bool = False
    use_srm_filters: bool = False

    def validate(self):
        """Validate configuration"""
        if self.num_classes <= 0:
           raise ValueError("num_classes must be positive")
        if not (0 <= self.dropout <= 1):
           raise ValueError(f"dropout must be in [0, 1]. Got: {self.dropout}")
        if self.freeze_epochs < 0:
           raise ValueError("freeze_epochs must be non-negative")



@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    # Training duration
    num_epochs: int = 50
    max_epochs: int = 200

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # 'adam', 'sgd', 'adamw'
    momentum: float = 0.9  # For SGD

    # Learning rate scheduling
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau', 'none'
    scheduler_patience: int = 5  # For ReduceLROnPlateau
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 10  # For StepLR
    warmup_epochs: int = 0

    # Loss function
    loss_function: str = "ce"  # 'ce', 'focal', 'weighted_ce', 'label_smoothing'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # Regularization
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = True

    # Advanced training
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    def validate(self):
        """Validate configuration"""
        if self.num_epochs <= 0:
           raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
           raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
           raise ValueError("weight_decay must be non-negative")
        if self.optimizer not in ['adam', 'sgd', 'adamw']:
           raise ValueError(f"optimizer must be one of: adam, sgd, adamw. Got: {self.optimizer}")
        if self.scheduler not in ['cosine', 'step', 'plateau', 'none']:
           raise ValueError(f"scheduler must be one of: cosine, step, plateau, none. Got: {self.scheduler}")
        if self.loss_function not in ['ce', 'focal', 'weighted_ce', 'label_smoothing']:
           raise ValueError(f"loss_function must be one of: ce, focal, weighted_ce, label_smoothing. Got: {self.loss_function}")


@dataclass
class InferenceConfig:
    """Inference configuration"""

    # Model
    model_path: str = "checkpoints/best_model.pth"

    # Inference settings
    batch_size: int = 64
    num_workers: int = 4
    use_amp: bool = True
    threshold: float = 0.5

    # Test-time augmentation
    use_tta: bool = False
    tta_crops: int = 5

    def validate(self):
        """Validate configuration"""
        if self.batch_size <= 0:
           raise ValueError("batch_size must be positive")
        if not (0 <= self.threshold <= 1):
           raise ValueError(f"threshold must be in [0, 1]. Got: {self.threshold}")

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""

    # Logging
    log_dir: str = "./logs"
    log_interval: int = 10  # Log every N batches

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 5  # Save every N epochs
    save_best_only: bool = True

    # Metrics tracking
    track_metrics: list = field(default_factory=lambda: ['accuracy', 'f1', 'auc'])
    primary_metric: str = 'f1'

    # Visualization
    save_plots: bool = True
    plot_dir: str = "./plots"

    def validate(self):
        """Validate configuration"""
        if self.log_interval <= 0:
           raise ValueError("log_interval must be positive")
        if self.save_frequency <= 0:
           raise ValueError("save_frequency must be positive")


@dataclass
class Config:
    """Main configuration class combining all sub-configs"""

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    device: str = "cuda"
    seed: int = 42
    experiment_name: str = "steg_exp"
    description: str = ""

    # Paths
    project_root: str = "."

    def validate(self):
        """Validate entire configuration"""
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.inference.validate()
        self.logging.validate()

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.logging.log_dir,
            self.logging.checkpoint_dir,
            self.logging.plot_dir,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML file

        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """
        Load configuration from JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            Config object
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create Config from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config object
        """
        config = cls()

        # Update sub-configs
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'inference' in config_dict:
            config.inference = InferenceConfig(**config_dict['inference'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])

        # Update global settings
        for key in ['device', 'seed', 'experiment_name', 'description', 'project_root']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary

        Returns:
            Configuration dictionary
        """
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'inference': asdict(self.inference),
            'logging': asdict(self.logging),
            'device': self.device,
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'project_root': self.project_root,
        }

    def save_yaml(self, path: str):
        """
        Save configuration to YAML file

        Args:
            path: Output path
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

        print(f"Configuration saved to {path_obj}")

    def save_json(self, path: str):
        """
        Save configuration to JSON file

        Args:
            path: Output path
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Configuration saved to {path_obj}")

    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)

        print(f"\nExperiment: {self.experiment_name}")
        if self.description:
            print(f"Description: {self.description}")

        print("\nData Configuration:")
        for key, value in asdict(self.data).items():
            print(f"  {key}: {value}")

        print("\nModel Configuration:")
        for key, value in asdict(self.model).items():
            print(f"  {key}: {value}")

        print("\nTraining Configuration:")
        for key, value in asdict(self.training).items():
            print(f"  {key}: {value}")

        print("\nGlobal Settings:")
        print(f"  device: {self.device}")
        print(f"  seed: {self.seed}")

        print("="*60 + "\n")


def load_config(config_path: str) -> Config:
    """
    Load configuration from file (auto-detects format)

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path_obj}")

    if config_path_obj.suffix in ['.yaml', '.yml']:
        return Config.from_yaml(str(config_path_obj))
    elif config_path_obj.suffix == '.json':
        return Config.from_json(str(config_path_obj))
    else:
        raise ValueError(f"Unsupported config format: {config_path_obj.suffix}")


def create_default_config(save_path: str = "config.yaml") -> Config:
    """
    Create and save default configuration

    Args:
        save_path: Path to save configuration

    Returns:
        Config object with defaults
    """
    config = Config()

    if save_path.endswith('.json'):
        config.save_json(save_path)
    else:
        config.save_yaml(save_path)

    return config


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge configuration with overrides

    Args:
        base_config: Base configuration
        override_dict: Dictionary of values to override

    Returns:
        Merged configuration
    """
    config_dict = base_config.to_dict()

    # Deep merge
    for key, value in override_dict.items():
        if key in config_dict and isinstance(config_dict[key], dict):
            config_dict[key].update(value)
        else:
            config_dict[key] = value

    return Config.from_dict(config_dict)


def config_from_args(args) -> Config:
    """
    Create configuration from command-line arguments

    Args:
        args: argparse.Namespace object

    Returns:
        Config object
    """
    # Start with default or load from file
    if hasattr(args, 'config') and args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Override with command-line arguments
    override_dict = {}

    if hasattr(args, 'batch_size'):
        override_dict.setdefault('data', {})['batch_size'] = args.batch_size

    if hasattr(args, 'learning_rate'):
        override_dict.setdefault('training', {})['learning_rate'] = args.learning_rate

    if hasattr(args, 'epochs'):
        override_dict.setdefault('training', {})['num_epochs'] = args.epochs

    if hasattr(args, 'model'):
        override_dict.setdefault('model', {})['model_name'] = args.model

    if override_dict:
        config = merge_configs(config, override_dict)

    return config


# Example usage functions
def get_training_config() -> Config:
    """Get a pre-configured training setup"""
    config = Config()
    config.experiment_name = "baseline_training"
    config.data.batch_size = 32
    config.training.num_epochs = 50
    config.training.learning_rate = 0.001
    config.model.model_name = "resnet50_steg"
    return config


def get_quick_test_config() -> Config:
    """Get a configuration for quick testing"""
    config = Config()
    config.experiment_name = "quick_test"
    config.data.batch_size = 16
    config.training.num_epochs = 5
    config.training.early_stopping_patience = 3
    return config


def get_production_config() -> Config:
    """Get a production-ready configuration"""
    config = Config()
    config.experiment_name = "production"
    config.data.batch_size = 64
    config.data.num_workers = 8
    config.training.num_epochs = 100
    config.training.use_mixed_precision = True
    config.logging.save_best_only = True
    return config
