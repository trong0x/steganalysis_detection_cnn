import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "steganalysis",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Logger specifically for training"""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        self.logger = setup_logger(
            name=f"training_{experiment_name}",
            log_file=str(log_file)
        )

        self.experiment_name = experiment_name
        self.start_time = None

    def log_config(self, config: dict):
        """Log configuration"""
        self.logger.info("="*60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("="*60)
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("="*60)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.logger.info(f"\nEpoch {epoch}/{total_epochs}")
        self.logger.info("-"*60)
        self.start_time = datetime.now()

    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log epoch end with metrics"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Epoch {epoch} completed in {elapsed:.2f}s")

        self.logger.info("Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("-"*60)

    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log when best model is saved"""
        self.logger.info(
            f"New best model saved at epoch {epoch} "
            f"({metric_name}: {metric_value:.4f})"
        )

    def log_training_complete(self, best_metrics: dict):
        """Log training completion"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Training completed!")
        self.logger.info("Best metrics:")
        for key, value in best_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("="*60)


class InferenceLogger:
    """Logger for inference"""

    def __init__(self, log_dir: str = "./logs/inference"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"inference_{timestamp}.log"

        self.logger = setup_logger(
            name="inference",
            log_file=str(log_file)
        )

    def log_prediction(self, file_path: str, prediction: dict):
        """Log single prediction"""
        self.logger.info(f"File: {file_path}")
        self.logger.info(f"  Prediction: {prediction['prediction']}")
        self.logger.info(f"  Confidence: {prediction['confidence']:.4f}")
        if 'inference_time' in prediction:
            self.logger.info(f"  Time: {prediction['inference_time']:.4f}s")

    def log_batch_summary(self, total: int, predictions: list):
        """Log batch prediction summary"""
        stego_count = sum(1 for p in predictions if p['prediction'] == 'Stego')
        cover_count = total - stego_count

        self.logger.info("\n" + "="*60)
        self.logger.info(f"Batch prediction summary:")
        self.logger.info(f"  Total files: {total}")
        self.logger.info(f"  Stego detected: {stego_count} ({stego_count/total*100:.2f}%)")
        self.logger.info(f"  Cover images: {cover_count} ({cover_count/total*100:.2f}%)")
        self.logger.info("="*60)


def get_logger(name: str = "steganalysis") -> logging.Logger:
    """Get or create a logger"""
    return logging.getLogger(name)
