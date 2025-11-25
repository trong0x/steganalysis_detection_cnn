
import torch
import torchvision.transforms as T
import torchaudio
from typing import Optional, List, Tuple
import random
import numpy as np
from PIL import Image, ImageFilter


class StegImageAugmentation:
    """
    Augmentation pipeline for steganalysis images
    Provides multiple augmentation strategies with different strengths
    """

    @staticmethod
    def get_strong_augmentation(img_size: int = 224) -> T.Compose:
        """
        Strong augmentation for training with limited data

        Args:
            img_size: Target image size

        Returns:
            Composed transforms with strong augmentation

        Use when:
        - You have limited training data
        - Model is overfitting
        - Need maximum generalization
        """
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])

    @staticmethod
    def get_medium_augmentation(img_size: int = 224) -> T.Compose:
        """
        Medium augmentation - balanced approach

        Args:
            img_size: Target image size

        Returns:
            Composed transforms with medium augmentation

        Recommended for:
        - Most steganalysis tasks
        - Balanced training
        - Good accuracy without over-augmentation
        """
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05)
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_light_augmentation(img_size: int = 224) -> T.Compose:
        """
        Light augmentation - minimal changes

        Args:
            img_size: Target image size

        Returns:
            Composed transforms with light augmentation

        Use when:
        - Steganographic signals are very subtle
        - You have lots of training data
        - Concerned about destroying hidden information
        """
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_test_augmentation(img_size: int = 224, n_crops: int = 5) -> T.Compose:
        """
        Test-time augmentation (TTA)

        Args:
            img_size: Target image size
            n_crops: Number of crops (5 or 10)

        Returns:
            Composed transforms for TTA

        Usage:
            Apply multiple augmented versions during inference
            and average the predictions for better accuracy
        """
        crop_transform = T.FiveCrop(img_size) if n_crops == 5 else T.TenCrop(img_size)

        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            crop_transform,
            T.Lambda(lambda crops: torch.stack([
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(T.ToTensor()(crop)) for crop in crops
            ]))
        ])

    @staticmethod
    def get_custom_augmentation(
        img_size: int = 224,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation: int = 10,
        color_jitter: bool = True,
        blur_prob: float = 0.0
    ) -> T.Compose:
        """
        Create custom augmentation pipeline

        Args:
            img_size: Target image size
            horizontal_flip: Enable horizontal flip
            vertical_flip: Enable vertical flip
            rotation: Rotation degrees (0 to disable)
            color_jitter: Enable color jittering
            blur_prob: Probability of applying Gaussian blur

        Returns:
            Custom composed transforms
        """
        transforms_list: List[object] = [T.Resize((img_size, img_size))]

        if horizontal_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))

        if vertical_flip:
            transforms_list.append(T.RandomVerticalFlip(p=0.5)) # type: ignore

        if rotation > 0:
            transforms_list.append(T.RandomRotation(degrees=rotation))

        if color_jitter:
            transforms_list.append(T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            )) # pyright: ignore[reportArgumentType]

        if blur_prob > 0:
            transforms_list.append(T.RandomApply(
                [T.GaussianBlur(kernel_size=3)],
                p=blur_prob
            ))

        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return T.Compose(transforms_list)




class AudioAugmentation:
    """Individual audio augmentation techniques"""

    @staticmethod
    def add_noise(
        waveform: torch.Tensor,
        noise_level: float = 0.005
    ) -> torch.Tensor:
        """
        Add Gaussian noise to waveform

        Args:
            waveform: Audio tensor (channels, samples)
            noise_level: Standard deviation of noise

        Returns:
            Noisy waveform
        """
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    @staticmethod
    def time_shift(
        waveform: torch.Tensor,
        max_shift: int = 1600
    ) -> torch.Tensor:
        """
        Randomly shift audio in time

        Args:
            waveform: Audio tensor
            max_shift: Maximum shift in samples

        Returns:
            Time-shifted waveform
        """
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(waveform, shifts=shift, dims=-1)

    @staticmethod
    def change_speed(
        waveform: torch.Tensor,
        speed_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Change playback speed

        Args:
            waveform: Audio tensor
            speed_factor: Speed multiplier (None = random 0.9-1.1)

        Returns:
            Speed-changed waveform
        """
        if speed_factor is None:
            speed_factor = 0.9 + random.random() * 0.2  # 0.9 to 1.1

        length = waveform.shape[-1]
        new_length = int(length / speed_factor)

        # Resample
        waveform_resampled = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)

        # Pad or truncate to original length
        if waveform_resampled.shape[-1] < length:
            padding = length - waveform_resampled.shape[-1]
            waveform_resampled = torch.nn.functional.pad(
                waveform_resampled, (0, padding)
            )
        else:
            waveform_resampled = waveform_resampled[..., :length]

        return waveform_resampled

    @staticmethod
    def time_mask(
        waveform: torch.Tensor,
        max_mask_size: int = 1600,
        num_masks: int = 1
    ) -> torch.Tensor:
        """
        Apply time masking

        Args:
            waveform: Audio tensor
            max_mask_size: Maximum mask size in samples
            num_masks: Number of masks to apply

        Returns:
            Masked waveform
        """
        length = waveform.shape[-1]
        masked = waveform.clone()

        for _ in range(num_masks):
            mask_size = random.randint(0, max_mask_size)
            if mask_size > 0 and length > mask_size:
                mask_start = random.randint(0, length - mask_size)
                masked[..., mask_start:mask_start + mask_size] = 0

        return masked

    @staticmethod
    def change_volume(
        waveform: torch.Tensor,
        gain_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Change volume

        Args:
            waveform: Audio tensor
            gain_db: Gain in decibels (None = random -3 to 3 dB)

        Returns:
            Volume-adjusted waveform
        """
        if gain_db is None:
            gain_db = random.uniform(-3, 3)

        gain_factor = 10 ** (gain_db / 20)
        return waveform * gain_factor

    @staticmethod
    def add_reverb(
        waveform: torch.Tensor,
        room_scale: float = 0.3
    ) -> torch.Tensor:
        """
        Add simple reverb effect

        Args:
            waveform: Audio tensor
            room_scale: Room size (0-1)

        Returns:
            Reverberated waveform
        """
        delay_samples = int(room_scale * 8000)  # Up to 0.5s at 16kHz
        decay = 0.5

        if delay_samples > 0 and delay_samples < waveform.shape[-1]:
            delayed = torch.roll(waveform, shifts=delay_samples, dims=-1)
            delayed[..., :delay_samples] = 0  # Zero out wrapped samples
            return waveform + decay * delayed

        return waveform


class StegAudioAugmentation:
    """
    Combined augmentation pipeline for audio steganalysis
    Applies multiple augmentations with controllable probabilities
    """

    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_level: float = 0.005,
        shift_prob: float = 0.5,
        max_shift: int = 1600,
        speed_prob: float = 0.3,
        mask_prob: float = 0.3,
        max_mask_size: int = 1600,
        volume_prob: float = 0.3,
        reverb_prob: float = 0.2
    ):
        """
        Args:
            noise_prob: Probability of adding noise
            noise_level: Noise standard deviation
            shift_prob: Probability of time shifting
            max_shift: Maximum time shift in samples
            speed_prob: Probability of speed change
            mask_prob: Probability of time masking
            max_mask_size: Maximum mask size
            volume_prob: Probability of volume change
            reverb_prob: Probability of adding reverb
        """
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self.speed_prob = speed_prob
        self.mask_prob = mask_prob
        self.max_mask_size = max_mask_size
        self.volume_prob = volume_prob
        self.reverb_prob = reverb_prob
        self.aug = AudioAugmentation()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations

        Args:
            waveform: Audio tensor (channels, samples)

        Returns:
            Augmented waveform
        """
        # Add noise
        if random.random() < self.noise_prob:
            waveform = self.aug.add_noise(waveform, self.noise_level)

        # Time shift
        if random.random() < self.shift_prob:
            waveform = self.aug.time_shift(waveform, self.max_shift)

        # Speed change
        if random.random() < self.speed_prob:
            waveform = self.aug.change_speed(waveform)

        # Time masking
        if random.random() < self.mask_prob:
            waveform = self.aug.time_mask(waveform, self.max_mask_size)

        # Volume change
        if random.random() < self.volume_prob:
            waveform = self.aug.change_volume(waveform)

        # Reverb
        if random.random() < self.reverb_prob:
            waveform = self.aug.add_reverb(waveform)

        return waveform


class SpecAugment:
    """
    SpecAugment for spectrograms (Park et al., 2019)
    Frequency and time masking for robust audio models
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
        """
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment

        Args:
            spectrogram: Spectrogram tensor (channels, freq, time)

        Returns:
            Augmented spectrogram
        """
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_mask(spectrogram)

        # Apply time masking
        for _ in range(self.num_time_masks):
            spectrogram = self.time_mask(spectrogram)

        return spectrogram


class MixupAugmentation:
    """
    Mixup augmentation (Zhang et al., 2017)
    Mixes two samples and their labels for regularization
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Mix two samples

        Args:
            x1: First sample
            x2: Second sample
            y1: First label
            y2: Second label

        Returns:
            (mixed_x, lambda, y1, y2)
            Final loss should be: lambda * loss(pred, y1) + (1-lambda) * loss(pred, y2)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        mixed_x = lam * x1 + (1 - lam) * x2

        return mixed_x, lam, y1, y2


class CutMixAugmentation:
    """
    CutMix augmentation (Yun et al., 2019)
    Cuts and pastes patches between images
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Apply CutMix

        Args:
            x1: First sample (B, C, H, W)
            x2: Second sample
            y1: First label
            y2: Second label

        Returns:
            (mixed_x, lambda, y1, y2)
        """
        lam = np.random.beta(self.alpha, self.alpha)

        _, _, H, W = x1.shape

        # Random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply patch
        x1_mixed = x1.clone()
        x1_mixed[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return x1_mixed, lam, y1, y2


class RandomAugmentationChoice:
    """
    Randomly select from multiple augmentation strategies
    Useful for exploring different augmentation combinations
    """

    def __init__(self, transforms: List):
        """
        Args:
            transforms: List of augmentation transforms to choose from
        """
        self.transforms = transforms

    def __call__(self, img):
        """Apply randomly selected transform"""
        transform = random.choice(self.transforms)
        return transform(img)


class AutoAugment:
    """
    AutoAugment for images
    Pre-defined policies that work well
    """

    @staticmethod
    def get_policy(img_size: int = 224) -> T.Compose:
        """
        Get AutoAugment policy

        Args:
            img_size: Target image size

        Returns:
            Composed transforms with AutoAugment
        """
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# Preset configurations for easy use
PRESET_AUGMENTATIONS = {
    'none': lambda size: T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'light': StegImageAugmentation.get_light_augmentation,
    'medium': StegImageAugmentation.get_medium_augmentation,
    'strong': StegImageAugmentation.get_strong_augmentation,
    'auto': AutoAugment.get_policy,
}


def get_augmentation(preset: str = 'medium', img_size: int = 224):
    """
    Get augmentation by preset name

    Args:
        preset: One of 'none', 'light', 'medium', 'strong', 'auto'
        img_size: Target image size

    Returns:
        Augmentation transform

    Example:
        >>> transform = get_augmentation('medium', 224)
        >>> augmented_img = transform(img)
    """
    if preset not in PRESET_AUGMENTATIONS:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available: {list(PRESET_AUGMENTATIONS.keys())}"
        )

    return PRESET_AUGMENTATIONS[preset](img_size)
