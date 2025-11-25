import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchaudio


# Helper to ensure returned images are torch tensors (C, H, W) float32 in [0, 1]
def _to_tensor(img) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        return img
    if isinstance(img, Image.Image):
        arr = np.array(img)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Convert dtype and normalize if coming from uint8 (0-255)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    tensor = torch.from_numpy(arr.copy())

    # Convert HWC -> CHW
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).contiguous()
    else:
        tensor = tensor.unsqueeze(0).contiguous()

    return tensor


class ImageStegDataset(Dataset):
    """
    Dataset for image steganalysis
    Loads cover (clean) and stego (hidden data) images
    """

    def __init__(
        self,
        cover_dir: str,
        stego_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        extensions: Optional[List[str]] = None,
    ):
        """
        Args:
            cover_dir: Directory containing cover (clean) images
            stego_dir: Directory containing stego (hidden data) images
            transform: Optional transforms to apply
            split: Dataset split ('train', 'val', 'test')
            extensions: List of valid file extensions
        """
        self.cover_dir = Path(cover_dir)
        self.stego_dir = Path(stego_dir)
        self.transform = transform
        self.split = split

        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        # Load image paths
        self.cover_paths = []
        self.stego_paths = []

        for ext in extensions:
            self.cover_paths.extend(sorted(self.cover_dir.glob(f"*{ext}")))
            self.cover_paths.extend(sorted(self.cover_dir.glob(f"*{ext.upper()}")))
            self.stego_paths.extend(sorted(self.stego_dir.glob(f"*{ext}")))
            self.stego_paths.extend(sorted(self.stego_dir.glob(f"*{ext.upper()}")))

        # Remove duplicates and sort
        self.cover_paths = sorted(list(set(self.cover_paths)))
        self.stego_paths = sorted(list(set(self.stego_paths)))

        # Create samples
        self.samples = [(str(p), 0) for p in self.cover_paths] + [
            (str(p), 1) for p in self.stego_paths
        ]

        if split == "train":
            random.shuffle(self.samples)

        print(f"[{split.upper()}] Loaded {len(self.cover_paths)} cover images")
        print(f"[{split.upper()}] Loaded {len(self.stego_paths)} stego images")
        print(f"[{split.upper()}] Total: {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Tuple[torch.Tensor, int]:
        """Get item with retry limit to prevent infinite loops"""
        MAX_RETRIES = 10

        if _retry_count >= MAX_RETRIES:
            raise RuntimeError(
                f"Failed to load valid sample after {MAX_RETRIES} attempts. "
                "Check your dataset for corrupt files."
            )

        img_path, label = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            # Ensure a torch.Tensor is returned
            image = _to_tensor(image)

            return image, label

        except Exception as e:
            print(f"Error loading {img_path}: {e} (retry {_retry_count + 1}/{MAX_RETRIES})")
            # Return a random valid sample instead, with retry counter
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx, _retry_count=_retry_count + 1)

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


class AudioStegDataset(Dataset):
    """
    Dataset for audio steganalysis
    Loads cover and stego audio files
    """

    def __init__(
        self,
        cover_dir: str,
        stego_dir: str,
        transform: Optional[Callable] = None,
        sample_rate: int = 16000,
        duration: float = 3.0,
        split: str = "train",
        extensions: Optional[List[str]] = None,
    ):
        """
        Args:
            cover_dir: Directory containing cover audio files
            stego_dir: Directory containing stego audio files
            transform: Optional transforms
            sample_rate: Target sample rate
            duration: Duration to load (seconds)
            split: Dataset split
            extensions: List of valid file extensions
        """
        self.cover_dir = Path(cover_dir)
        self.stego_dir = Path(stego_dir)
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.split = split
        self.max_length = int(sample_rate * duration)

        if extensions is None:
            extensions = [".wav", ".mp3", ".flac", ".ogg"]

        # Load audio paths
        self.cover_paths = []
        self.stego_paths = []

        for ext in extensions:
            self.cover_paths.extend(sorted(self.cover_dir.glob(f"*{ext}")))
            self.cover_paths.extend(sorted(self.cover_dir.glob(f"*{ext.upper()}")))
            self.stego_paths.extend(sorted(self.stego_dir.glob(f"*{ext}")))
            self.stego_paths.extend(sorted(self.stego_dir.glob(f"*{ext.upper()}")))

        # Remove duplicates and sort
        self.cover_paths = sorted(list(set(self.cover_paths)))
        self.stego_paths = sorted(list(set(self.stego_paths)))

        # Create samples
        self.samples = [(str(p), 0) for p in self.cover_paths] + [
            (str(p), 1) for p in self.stego_paths
        ]

        if split == "train":
            random.shuffle(self.samples)

        print(f"[{split.upper()}] Loaded {len(self.cover_paths)} cover audio files")
        print(f"[{split.upper()}] Loaded {len(self.stego_paths)} stego audio files")
        print(f"[{split.upper()}] Total: {len(self.samples)} audio files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Tuple[torch.Tensor, int]:
        """Get item with retry limit to prevent infinite loops"""
        MAX_RETRIES = 10

        if _retry_count >= MAX_RETRIES:
            raise RuntimeError(
                f"Failed to load valid audio sample after {MAX_RETRIES} attempts. "
                "Check your dataset for corrupt files."
            )

        audio_path, label = self.samples[idx]

        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or truncate to fixed length
            if waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, : self.max_length]

            # Apply transforms
            if self.transform:
                waveform = self.transform(waveform)

            return waveform, label

        except Exception as e:
            print(f"Error loading {audio_path}: {e} (retry {_retry_count + 1}/{MAX_RETRIES})")
            # Return a random valid sample
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx, _retry_count=_retry_count + 1)


class PairedImageDataset(Dataset):
    """
    Dataset for paired cover-stego images
    Useful for training networks that compare pairs
    """

    def __init__(
        self,
        cover_dir: str,
        stego_dir: str,
        transform: Optional[Callable] = None,
        match_by_name: bool = True,
        extensions: Optional[List[str]] = None,
    ):
        self.cover_dir = Path(cover_dir)
        self.stego_dir = Path(stego_dir)
        self.transform = transform
        self.match_by_name = match_by_name

        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        cover_files = []
        stego_files = []
        for ext in extensions:
            cover_files.extend(sorted(self.cover_dir.glob(f"*{ext}")))
            cover_files.extend(sorted(self.cover_dir.glob(f"*{ext.upper()}")))
            stego_files.extend(sorted(self.stego_dir.glob(f"*{ext}")))
            stego_files.extend(sorted(self.stego_dir.glob(f"*{ext.upper()}")))

        cover_files = sorted(list(set(cover_files)))
        stego_files = sorted(list(set(stego_files)))

        if match_by_name:
            # Match by filename
            cover_dict = {p.stem: p for p in cover_files}
            stego_dict = {p.stem: p for p in stego_files}

            common_names = set(cover_dict.keys()) & set(stego_dict.keys())
            self.pairs = [(cover_dict[name], stego_dict[name]) for name in sorted(common_names)]
        else:
            # Pair by order
            min_len = min(len(cover_files), len(stego_files))
            self.pairs = list(zip(cover_files[:min_len], stego_files[:min_len]))

        print(f"Loaded {len(self.pairs)} cover-stego pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cover_path, stego_path = self.pairs[idx]

        cover_img = Image.open(cover_path).convert("RGB")
        stego_img = Image.open(stego_path).convert("RGB")

        if self.transform:
            cover_img = self.transform(cover_img)
            stego_img = self.transform(stego_img)

        # Ensure both images are torch.Tensor (C, H, W)
        cover_img = _to_tensor(cover_img)
        stego_img = _to_tensor(stego_img)

        return cover_img, stego_img


class InMemoryDataset(Dataset):
    """
    Dataset that loads all images into memory for faster training
    """

    def __init__(
        self,
        cover_dir: str,
        stego_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        extensions: Optional[List[str]] = None,
    ):
        self.transform = transform
        self.split = split

        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        # Load all images into memory
        self.images: List[Image.Image] = []
        self.labels: List[int] = []

        print(f"Loading {split} dataset into memory...")

        # Load cover images
        cover_paths = []
        stego_paths = []
        for ext in extensions:
            cover_paths.extend(sorted(Path(cover_dir).glob(f"*{ext}")))
            cover_paths.extend(sorted(Path(cover_dir).glob(f"*{ext.upper()}")))
            stego_paths.extend(sorted(Path(stego_dir).glob(f"*{ext}")))
            stego_paths.extend(sorted(Path(stego_dir).glob(f"*{ext.upper()}")))

        cover_paths = sorted(list(set(cover_paths)))
        stego_paths = sorted(list(set(stego_paths)))

        for path in cover_paths:
            img = Image.open(path).convert("RGB")
            self.images.append(img)
            self.labels.append(0)

        for path in stego_paths:
            img = Image.open(path).convert("RGB")
            self.images.append(img)
            self.labels.append(1)

        print(f"Loaded {len(self.images)} images into memory")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert to tensor if needed to satisfy return type
        image = _to_tensor(image)

        return image, label
