import torch
import torchaudio
import torchvision.transforms as T
from typing import Tuple, Optional
import numpy as np


class ImagePreprocessor:
    """Preprocessing pipeline for images"""

    @staticmethod
    def get_train_transforms(
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> T.Compose:
        """
        Get training transforms with basic augmentation

        Args:
            img_size: Target image size
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)

        Returns:
            Composed transforms
        """
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    @staticmethod
    def get_val_transforms(
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> T.Compose:
        """
        Get validation/test transforms without augmentation

        Args:
            img_size: Target image size
            mean: Normalization mean
            std: Normalization std

        Returns:
            Composed transforms
        """
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    @staticmethod
    def denormalize(
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> torch.Tensor:
        """
        Denormalize image tensor for visualization

        Args:
            tensor: Normalized tensor (C, H, W)
            mean: Mean used for normalization
            std: Std used for normalization

        Returns:
            Denormalized tensor
        """
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return tensor * std_tensor + mean_tensor

    @staticmethod
    def resize_with_aspect_ratio(
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> T.Compose:
        """
        Resize while maintaining aspect ratio with padding

        Args:
            img_size: Target size
            mean: Normalization mean
            std: Normalization std

        Returns:
            Composed transforms
        """
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])


class AudioPreprocessor:
    """Preprocessing pipeline for audio"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        n_mels: int = 128,
        hop_length: int = 256,
        n_mfcc: Optional[int] = None
    ):
        """
        Args:
            sample_rate: Target sample rate
            n_fft: FFT size
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
            n_mfcc: Number of MFCC coefficients (None = use mel spectrogram)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

        if n_mfcc is None:
            # Mel spectrogram transform
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=hop_length
            )
            self.use_mfcc = False
        else:
            # MFCC transform
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'n_mels': n_mels
                }
            )
            self.use_mfcc = True

        # Convert to dB scale
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to spectrogram

        Args:
            waveform: Audio waveform tensor (channels, samples)

        Returns:
            Spectrogram tensor
        """
        # Compute spectrogram
        spec = self.transform(waveform)

        # Convert to dB if not using MFCC
        if not self.use_mfcc:
            spec = self.amplitude_to_db(spec)

        # Normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        return spec

    @staticmethod
    def get_mfcc_transform(
        sample_rate: int = 16000,
        n_mfcc: int = 40
    ) -> torchaudio.transforms.MFCC:
        """
        Get MFCC transform

        Args:
            sample_rate: Sample rate
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC transform
        """
        return torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 512,
                'hop_length': 256,
                'n_mels': 128
            }
        )


class SpectrogramPreprocessor:
    """Advanced spectrogram preprocessing for audio"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        target_size: Tuple[int, int] = (224, 224),
        to_rgb: bool = True
    ):
        """
        Args:
            sample_rate: Sample rate
            n_fft: FFT size
            hop_length: Hop length
            target_size: Target image size (H, W)
            to_rgb: Convert to 3-channel RGB-like representation
        """
        self.sample_rate = sample_rate
        self.target_size = target_size
        self.to_rgb = to_rgb

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to spectrogram image

        Args:
            waveform: Audio waveform (channels, samples)

        Returns:
            Spectrogram as image tensor (C, H, W)
        """
        # Compute spectrogram
        spec = self.spectrogram(waveform)

        # Convert to dB
        spec_db = self.amplitude_to_db(spec)

        # Normalize to [0, 1]
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)

        # Resize to target size
        if spec_db.shape[-2:] != self.target_size:
            spec_db = torch.nn.functional.interpolate(
                spec_db.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Convert to RGB if requested
        if self.to_rgb and spec_db.shape[0] == 1:
            spec_db = spec_db.repeat(3, 1, 1)

        return spec_db


class WaveformPreprocessor:
    """
    Direct waveform preprocessing (for 1D CNNs)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        normalize: bool = True
    ):
        """
        Args:
            sample_rate: Target sample rate
            duration: Fixed duration in seconds
            normalize: Whether to normalize waveform
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.normalize = normalize
        self.max_length = int(sample_rate * duration)

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Preprocess waveform

        Args:
            waveform: Audio waveform
            sr: Original sample rate

        Returns:
            Preprocessed waveform
        """
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate
        if waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_length]

        # Normalize
        if self.normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        return waveform


class NoisePreprocessor:
    """
    Extract noise residual from images
    Useful for detecting subtle steganographic changes
    """

    def __init__(self, kernel_type: str = 'high_pass'):
        """
        Args:
            kernel_type: 'high_pass', 'laplacian', or 'sobel'
        """
        self.kernel_type = kernel_type

        if kernel_type == 'high_pass':
            # High-pass filter
            self.kernel = torch.FloatTensor([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ]) / 8.0
        elif kernel_type == 'laplacian':
            self.kernel = torch.FloatTensor([
                [0,  1, 0],
                [1, -4, 1],
                [0,  1, 0]
            ])
        elif kernel_type == 'sobel':
            # Sobel X
            self.kernel = torch.FloatTensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])

        # Prepare for conv2d
        self.kernel = self.kernel.view(1, 1, 3, 3)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract noise from image

        Args:
            image: Image tensor (C, H, W) or (B, C, H, W)

        Returns:
            Noise residual
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Apply to each channel
        noise = []
        for c in range(image.shape[1]):
            channel = image[:, c:c+1, :, :]
            noise_channel = torch.nn.functional.conv2d(
                channel,
                self.kernel.to(image.device),
                padding=1
            )
            noise.append(noise_channel)

        noise = torch.cat(noise, dim=1)

        if squeeze:
            noise = noise.squeeze(0)

        return noise


class MultiScalePreprocessor:
    """
    Create multi-scale representations of images
    """

    def __init__(self, scales: list = [224, 112, 56]):
        """
        Args:
            scales: List of target sizes
        """
        self.scales = scales
        self.transforms = [
            T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]) for size in scales
        ]

    def __call__(self, image):
        """
        Create multi-scale versions

        Returns:
            List of scaled images
        """
        return [transform(image) for transform in self.transforms]
