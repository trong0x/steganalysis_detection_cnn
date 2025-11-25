"""
Extract audio features for traditional ML approaches
"""

import numpy as np
import librosa
import torchaudio
from scipy import stats, signal
from typing import Dict, List, Tuple
import torch


class AudioFeatureExtractor:
    """Extract handcrafted features from audio"""

    @staticmethod
    def extract_mfcc_features(
        audio_path: str,
        sr: int = 16000,
        n_mfcc: int = 40
    ) -> Dict[str, np.ndarray]:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients)

        Args:
            audio_path: Path to audio file
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients

        Returns:
            Dictionary of MFCC features
        """
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        features = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'mfcc_max': np.max(mfccs, axis=1),
            'mfcc_min': np.min(mfccs, axis=1),
        }

        # Delta and delta-delta MFCCs
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)

        return features

    @staticmethod
    def extract_spectral_features(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract spectral features"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        features = {}

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)

        return features

    @staticmethod
    def extract_rhythm_features(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract rhythm and tempo features"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        features = {}

        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['num_beats'] = len(beats)

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)

        # Tempogram
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features['tempogram_mean'] = np.mean(tempogram)
        features['tempogram_std'] = np.std(tempogram)

        return features

    @staticmethod
    def extract_energy_features(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract energy-related features"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        features = {}

        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)

        # Total energy
        features['total_energy'] = np.sum(y**2)

        # Energy entropy
        frame_energies = librosa.feature.rms(y=y)[0]
        frame_energies_normalized = frame_energies / np.sum(frame_energies)
        features['energy_entropy'] = stats.entropy(frame_energies_normalized + 1e-10)

        return features

    @staticmethod
    def extract_chroma_features(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract chroma features (pitch class profiles)"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        features = {
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma),
            'chroma_max': np.max(chroma),
            'chroma_min': np.min(chroma),
        }

        # Chroma deviation (how much it varies)
        features['chroma_deviation'] = np.mean(np.std(chroma, axis=1))

        return features

    @staticmethod
    def extract_lsb_audio_features(
        audio_path: str,
        sr: int = 16000,
        bit_depth: int = 16
    ) -> Dict[str, float]:
        """
        Extract LSB features from audio (steganography often uses LSB)

        Args:
            audio_path: Path to audio file
            sr: Sample rate
            bit_depth: Bit depth (usually 16)

        Returns:
            LSB-related features
        """
        # Load audio
        waveform, _ = torchaudio.load(audio_path)

        # Convert to integer representation
        if bit_depth == 16:
            waveform_int = (waveform * 32767).long()
        else:
            waveform_int = (waveform * (2**(bit_depth-1) - 1)).long()

        # Extract LSBs
        lsb = (waveform_int & 1).float()

        features = {
            'lsb_mean': lsb.mean().item(),
            'lsb_std': lsb.std().item(),
            'lsb_entropy': stats.entropy(
                np.histogram(lsb.numpy().flatten(), bins=2)[0] + 1e-10
            ),
        }

        # LSB transition rate (how often LSB changes)
        lsb_diff = torch.abs(lsb[:, 1:] - lsb[:, :-1])
        features['lsb_transition_rate'] = lsb_diff.mean().item()

        return features

    @staticmethod
    def extract_noise_features(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract noise characteristics"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        features = {}

        # High-pass filter to isolate noise
        sos = signal.butter(5, 2000, 'hp', fs=sr, output='sos')
        high_freq = signal.sosfilt(sos, y)

        # Ensure a float numpy array for numeric ops (fixes type checking issues)
        high_freq = np.asarray(high_freq, dtype=np.float64)

        # Flatten multi-channel arrays to a single dimension for statistics
        high_freq = high_freq.flatten()

        features['high_freq_energy'] = float(np.sum(np.square(high_freq)))
        features['high_freq_mean'] = float(np.mean(np.abs(high_freq)))
        features['high_freq_std'] = float(np.std(high_freq))

        # Signal-to-noise ratio estimate
        signal_power = np.mean(np.square(y))
        noise_power = np.mean(np.square(high_freq))
        features['snr_estimate'] = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return features

    @staticmethod
    def extract_statistical_features(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract statistical features from waveform"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        features = {
            'mean': np.mean(y),
            'std': np.std(y),
            'median': np.median(y),
            'max': np.max(y),
            'min': np.min(y),
            'skewness': stats.skew(y),
            'kurtosis': stats.kurtosis(y),
            'rms': np.sqrt(np.mean(y**2)),
        }

        # Dynamic range
        features['dynamic_range'] = np.max(np.abs(y)) - np.min(np.abs(y))

        # Peak-to-average ratio
        features['peak_to_avg_ratio'] = np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)

        return features

    @staticmethod
    def extract_all_features(audio_path: str, sr: int = 16000) -> Dict[str, float]:
        """
        Extract all audio features

        Args:
            audio_path: Path to audio file
            sr: Sample rate

        Returns:
            Dictionary of all features (flattened)
        """
        all_features = {}

        # Extract all feature types
        mfcc_features = AudioFeatureExtractor.extract_mfcc_features(audio_path, sr)
        spectral_features = AudioFeatureExtractor.extract_spectral_features(audio_path, sr)
        rhythm_features = AudioFeatureExtractor.extract_rhythm_features(audio_path, sr)
        energy_features = AudioFeatureExtractor.extract_energy_features(audio_path, sr)
        chroma_features = AudioFeatureExtractor.extract_chroma_features(audio_path, sr)
        lsb_features = AudioFeatureExtractor.extract_lsb_audio_features(audio_path, sr)
        noise_features = AudioFeatureExtractor.extract_noise_features(audio_path, sr)
        stat_features = AudioFeatureExtractor.extract_statistical_features(audio_path, sr)

        # Flatten MFCC arrays
        for key, value in mfcc_features.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    all_features[f'{key}_{i}'] = v
            else:
                all_features[key] = value

        # Add other features
        all_features.update(spectral_features)
        all_features.update(rhythm_features)
        all_features.update(energy_features)
        all_features.update(chroma_features)
        all_features.update(lsb_features)
        all_features.update(noise_features)
        all_features.update(stat_features)

        return all_features

    @staticmethod
    def extract_features_batch(
        audio_paths: List[str],
        sr: int = 16000
    ) -> np.ndarray:
        """
        Extract features from multiple audio files

        Args:
            audio_paths: List of audio file paths
            sr: Sample rate

        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = []

        for path in audio_paths:
            try:
                features = AudioFeatureExtractor.extract_all_features(path, sr)
                # Convert to list maintaining order
                feature_vector = [features[k] for k in sorted(features.keys())]
                all_features.append(feature_vector)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Add zeros for failed files
                if all_features:
                    all_features.append([0] * len(all_features[0]))

        return np.array(all_features)


class SpectrogramFeatureExtractor:
    """Extract features from spectrograms"""

    @staticmethod
    def extract_spectrogram_statistics(
        audio_path: str,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Extract statistical features from spectrogram"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Compute spectrogram
        D = np.abs(librosa.stft(y))

        features = {
            'spec_mean': np.mean(D),
            'spec_std': np.std(D),
            'spec_max': np.max(D),
            'spec_min': np.min(D),
            'spec_median': np.median(D),
        }

        # Frequency distribution
        freq_distribution = np.mean(D, axis=1)
        features['freq_dist_mean'] = np.mean(freq_distribution)
        features['freq_dist_std'] = np.std(freq_distribution)

        # Temporal distribution
        temporal_distribution = np.mean(D, axis=0)
        features['temporal_dist_mean'] = np.mean(temporal_distribution)
        features['temporal_dist_std'] = np.std(temporal_distribution)

        return features
