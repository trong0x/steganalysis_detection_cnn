import numpy as np
from PIL import Image
import cv2
from scipy import stats, fftpack
from typing import Dict, List
import torch


class ImageFeatureExtractor:
    """Extract handcrafted features from images for steganalysis"""

    @staticmethod
    def extract_color_statistics(image: np.ndarray) -> Dict[str, float]:
        """
        Extract color channel statistics

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Dictionary of color statistics
        """
        features = {}

        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i].flatten()

            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
            features[f'{channel}_median'] = float(np.median(channel_data))
            features[f'{channel}_min'] = float(np.min(channel_data))
            features[f'{channel}_max'] = float(np.max(channel_data))
            features[f'{channel}_skewness'] = float(stats.skew(channel_data))
            features[f'{channel}_kurtosis'] = float(stats.kurtosis(channel_data))
            features[f'{channel}_variance'] = float(np.var(channel_data))

        return features

    @staticmethod
    def extract_histogram_features(
        image: np.ndarray,
        bins: int = 256
    ) -> Dict[str, float]:
        """Extract histogram-based features"""
        features = {}

        for i, channel in enumerate(['R', 'G', 'B']):
            hist, _ = np.histogram(
                image[:, :, i],
                bins=bins,
                range=(0, 256)
            )
            hist = hist / np.sum(hist)  # Normalize

            # Entropy
            features[f'{channel}_entropy'] = float(
                stats.entropy(hist + 1e-10)
            )

            # Mode
            features[f'{channel}_mode'] = float(np.argmax(hist))

            # Histogram spread
            features[f'{channel}_hist_spread'] = float(
                np.sum(hist * np.arange(bins))
            )

        return features

    @staticmethod
    def extract_texture_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using gradients

        Args:
            image: RGB or grayscale image

        Returns:
            Texture features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float64)

        # Calculate gradients
        grad_x = np.asarray(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).astype(np.float64)
        grad_y = np.asarray(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).astype(np.float64)

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features = {
            'gradient_magnitude_mean': float(np.mean(gradient_magnitude)),
            'gradient_magnitude_std': float(np.std(gradient_magnitude)),
            'gradient_magnitude_max': float(np.max(gradient_magnitude)),
            'gradient_x_mean': float(np.mean(np.abs(grad_x))),
            'gradient_y_mean': float(np.mean(np.abs(grad_y))),
            'gradient_x_std': float(np.std(grad_x)),
            'gradient_y_std': float(np.std(grad_y)),
        }

        # Laplacian (edge detection)
        laplacian = np.asarray(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float64)
        features['laplacian_mean'] = float(np.mean(np.abs(laplacian)))
        features['laplacian_std'] = float(np.std(laplacian))
        features['laplacian_energy'] = float(np.sum(laplacian**2))

        # Canny edges
        edges = np.asarray(cv2.Canny(gray.astype(np.uint8), 50, 150))
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)

        return features

    @staticmethod
    def extract_noise_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract noise-related features
        High-pass filter reveals noise characteristics
        """
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float64)

        # High-pass filter kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float64) / 8.0

        # Apply filter
        noise_residual = np.asarray(cv2.filter2D(gray, -1, kernel)).astype(np.float64)

        features['noise_mean'] = float(np.mean(np.abs(noise_residual)))
        features['noise_std'] = float(np.std(noise_residual))
        features['noise_max'] = float(np.max(np.abs(noise_residual)))
        features['noise_energy'] = float(np.sum(noise_residual**2))
        features['noise_entropy'] = float(
            stats.entropy(
                np.histogram(noise_residual.flatten(), bins=50)[0] + 1e-10
            )
        )

        return features

    @staticmethod
    def extract_frequency_features(image: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT"""
        features = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Frequency statistics
        features['fft_mean'] = float(np.mean(magnitude))
        features['fft_std'] = float(np.std(magnitude))
        features['fft_max'] = float(np.max(magnitude))
        features['fft_energy'] = float(np.sum(magnitude**2))

        # High frequency energy ratio
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4

        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 > radius**2

        high_freq_energy = np.sum(magnitude[mask]**2)
        total_energy = np.sum(magnitude**2)

        features['high_freq_ratio'] = float(
            high_freq_energy / (total_energy + 1e-10)
        )
        features['low_freq_ratio'] = float(
            1.0 - features['high_freq_ratio']
        )

        return features

    @staticmethod
    def extract_lsb_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract Least Significant Bit (LSB) features
        Steganography often modifies LSBs
        """
        features = {}

        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i]

            # Extract LSB plane
            lsb_plane = channel_data & 1

            # LSB statistics
            features[f'{channel}_lsb_mean'] = float(np.mean(lsb_plane))
            features[f'{channel}_lsb_std'] = float(np.std(lsb_plane))

            # Bit plane randomness (entropy)
            hist, _ = np.histogram(lsb_plane.flatten(), bins=2, range=(0, 2))
            hist = hist / np.sum(hist)
            features[f'{channel}_lsb_entropy'] = float(
                stats.entropy(hist + 1e-10)
            )

            # LSB transitions (how often bit changes)
            lsb_flat = lsb_plane.flatten()
            transitions = np.sum(np.abs(lsb_flat[1:] - lsb_flat[:-1]))
            features[f'{channel}_lsb_transitions'] = float(
                transitions / len(lsb_flat)
            )

        return features

    @staticmethod
    def extract_co_occurrence_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract Gray Level Co-occurrence Matrix (GLCM) features
        Measures texture patterns
        """
        from skimage.feature import graycomatrix, graycoprops

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(
            gray,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        features = {}

        # Extract properties
        props = ['contrast', 'dissimilarity', 'homogeneity',
                'energy', 'correlation', 'ASM']

        for prop in props:
            values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = float(np.mean(values))
            features[f'glcm_{prop}_std'] = float(np.std(values))

        return features

    @staticmethod
    def extract_all_features(image_path: str) -> Dict[str, float]:
        """
        Extract all features from an image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary of all extracted features
        """
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))

        # Extract all feature types
        features = {}

        try:
            features.update(
                ImageFeatureExtractor.extract_color_statistics(image)
            )
        except Exception as e:
            print(f"Error extracting color stats: {e}")

        try:
            features.update(
                ImageFeatureExtractor.extract_histogram_features(image)
            )
        except Exception as e:
            print(f"Error extracting histogram features: {e}")

        try:
            features.update(
                ImageFeatureExtractor.extract_texture_features(image)
            )
        except Exception as e:
            print(f"Error extracting texture features: {e}")

        try:
            features.update(
                ImageFeatureExtractor.extract_noise_features(image)
            )
        except Exception as e:
            print(f"Error extracting noise features: {e}")

        try:
            features.update(
                ImageFeatureExtractor.extract_frequency_features(image)
            )
        except Exception as e:
            print(f"Error extracting frequency features: {e}")

        try:
            features.update(
                ImageFeatureExtractor.extract_lsb_features(image)
            )
        except Exception as e:
            print(f"Error extracting LSB features: {e}")

        try:
            features.update(
                ImageFeatureExtractor.extract_co_occurrence_features(image)
            )
        except Exception as e:
            print(f"Error extracting co-occurrence features: {e}")

        return features

    @staticmethod
    def extract_features_batch(image_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple images

        Args:
            image_paths: List of image paths

        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = []

        for i, path in enumerate(image_paths):
            try:
                features = ImageFeatureExtractor.extract_all_features(path)
                # Convert to list maintaining order
                if i == 0:
                    # First iteration - establish key order
                    key_order = sorted(features.keys())
                feature_vector = [features[k] for k in key_order]
                all_features.append(feature_vector)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Skip failed files
                continue

        return np.array(all_features)


class SRMFeatureExtractor:
    """
    Spatial Rich Model (SRM) feature extractor
    State-of-the-art handcrafted features for steganalysis
    """

    @staticmethod
    def get_srm_kernels() -> List[np.ndarray]:
        """Get SRM filter kernels"""
        kernels = []

        # Basic first-order filters
        kernels.append(np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64))

        kernels.append(np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64))

        kernels.append(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64))

        return kernels

    @staticmethod
    def apply_srm_filters(image: np.ndarray) -> np.ndarray:
        """Apply SRM filter bank"""

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.float64)

        kernels = SRMFeatureExtractor.get_srm_kernels()

        responses = []
        for kernel in kernels:
            response = cv2.filter2D(image, -1, kernel)
            responses.append(response)

        return np.stack(responses, axis=0)

    @staticmethod
    def extract_srm_features(image_path: str) -> Dict[str, float]:
        """Extract SRM-based features"""
        image = np.array(Image.open(image_path).convert('RGB'))

        # Apply SRM filters
        responses = SRMFeatureExtractor.apply_srm_filters(image)

        # Extract statistics from responses
        features = {}
        for i, response in enumerate(responses):
            features[f'srm_{i}_mean'] = float(np.mean(np.abs(response)))
            features[f'srm_{i}_std'] = float(np.std(response))
            features[f'srm_{i}_energy'] = float(np.sum(response**2))
            features[f'srm_{i}_max'] = float(np.max(np.abs(response)))

        return features
