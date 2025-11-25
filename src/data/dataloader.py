

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional
import numpy as np


class StegDataLoaderFactory:
    """Factory for creating dataloaders with proper configurations"""

    @staticmethod
    def create_dataloader(
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        use_balanced_sampling: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader with appropriate settings

        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            use_balanced_sampling: Use weighted sampling for class balance
        """
        sampler = None

        if use_balanced_sampling:
            # Calculate class weights
            # Ensure labels are Python ints so numpy operations behave as expected
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
            class_counts = np.bincount(labels)
            # Avoid division by zero for any class with zero count
            class_counts = np.where(class_counts == 0, 1, class_counts)
            class_weights = 1.0 / class_counts.astype(np.float64)
            # Convert weights to plain Python floats for WeightedRandomSampler
            sample_weights = [float(class_weights[label]) for label in labels]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # Sampler takes care of randomization

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler
        )

    @staticmethod
    def create_train_dataloader(
        dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        use_balanced_sampling: bool = False
    ) -> DataLoader:
        """Create training dataloader"""
        return StegDataLoaderFactory.create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            use_balanced_sampling=use_balanced_sampling
        )

    @staticmethod
    def create_val_dataloader(
        dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> DataLoader:
        """Create validation dataloader"""
        return StegDataLoaderFactory.create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    @staticmethod
    def create_test_dataloader(
        dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> DataLoader:
        """Create test dataloader"""
        return StegDataLoaderFactory.create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )


def collate_fn_image(batch):
    """Custom collate function for image batches"""
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def collate_fn_audio(batch):
    """Custom collate function for audio batches"""
    waveforms, labels = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels


class InfiniteDataLoader:
    """Wrapper for infinite data loading during training"""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
