import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchaudio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import json
import time


class ImagePathDataset(Dataset):
    """Dataset for loading images from paths"""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, path


class BatchPredictor:
    """Batch prediction for multiple files"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        batch_size: int = 32,
        num_workers: int = 4,
        use_amp: bool = True,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained model
            device: Device to use
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            use_amp: Use automatic mixed precision
            class_names: Class names for predictions
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp and device == 'cuda'
        self.class_names = class_names or ['Cover', 'Stego']

    @torch.no_grad()
    def predict_images(
        self,
        image_paths: List[str],
        transform,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict multiple images

        Args:
            image_paths: List of image file paths
            transform: Transform to apply to images
            show_progress: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        # Create dataset and dataloader
        dataset = ImagePathDataset(image_paths, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        results = []

        # Progress bar
        if show_progress:
            pbar = tqdm(dataloader, desc='Predicting images', total=len(dataloader))
        else:
            pbar = dataloader

        for images, paths in pbar:
            # Skip None values (failed loads)
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            if not valid_indices:
                # All failed
                for path in paths:
                    results.append({
                        'file_path': path,
                        'error': 'Failed to load image'
                    })
                continue

            valid_images = [images[i] for i in valid_indices]
            valid_paths = [paths[i] for i in valid_indices]

            # Stack valid images
            images_batch = torch.stack(valid_images).to(self.device)

            # Predict
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images_batch)
            else:
                outputs = self.model(images_batch)

            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = probs.max(1)

            # Store results
            for path, pred, conf, prob in zip(
                valid_paths,
                predictions.cpu().numpy(),
                confidences.cpu().numpy(),
                probs.cpu().numpy()
            ):
                results.append({
                    'file_path': path,
                    'prediction': self.class_names[pred],
                    'predicted_class': int(pred),
                    'confidence': float(conf),
                    'class_probabilities': {
                        self.class_names[i]: float(prob[i])
                        for i in range(len(self.class_names))
                    }
                })

            # Add failed loads
            for i, path in enumerate(paths):
                if i not in valid_indices:
                    results.append({
                        'file_path': path,
                        'error': 'Failed to load image'
                    })

        return results

    @torch.no_grad()
    def predict_audios(
        self,
        audio_paths: List[str],
        transform,
        sample_rate: int = 16000,
        duration: float = 3.0,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict multiple audio files
        """
        results = []
        iterator = tqdm(audio_paths, desc="Predicting audio files") if show_progress else audio_paths
        target_length = int(sample_rate * duration)

        for audio_path in iterator:
            try:
                # 1. Load
                waveform, orig_sr = torchaudio.load(audio_path)

                # 2. Resample
                if orig_sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_sr, sample_rate, dtype=waveform.dtype)
                    waveform = resampler(waveform)

                # 3. Mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # 4. Pad/truncate
                if waveform.shape[1] < target_length:
                    waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
                else:
                    waveform = waveform[:, :target_length]

                # 5. Transform (MelSpec → Log → etc.)
                if transform is not None:
                    waveform = transform(waveform)

                # 6. Add batch dim
                waveform = waveform.unsqueeze(0).to(self.device)

                # 7. Inference
                if self.use_amp and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(waveform)
                else:
                    outputs = self.model(waveform)  # ← chỉ 1 dòng

                # Handle InceptionV3 aux logits
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # 8. Softmax + confidence
                probs = torch.softmax(outputs, dim=1).squeeze(0)
                confidence, pred_idx = torch.max(probs, dim=0)
                pred_idx = int(pred_idx.item())
                conf = float(confidence.item())

                results.append({
                    "file_path": audio_path,
                    "prediction": self.class_names[pred_idx],
                    "predicted_class": pred_idx,
                    "confidence": conf,
                    "class_probabilities": {
                        self.class_names[i]: float(probs[i].item())
                        for i in range(len(self.class_names))
                    },
                })

            except Exception as e:
                results.append({
                    "file_path": audio_path,
                    "error": str(e)
                })

        return results

    def _collate_fn(self, batch):
        """Custom collate function to handle None values"""
        images, paths = zip(*batch)
        return list(images), list(paths)

    def predict_directory(
        self,
        directory: str,
        transform,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        modality: str = 'image'
    ) -> List[Dict]:
        """
        Predict all files in a directory

        Args:
            directory: Directory path
            transform: Transform to apply
            file_extensions: List of valid extensions
            recursive: Search recursively
            modality: 'image' or 'audio'

        Returns:
            List of predictions
        """
        dir_path = Path(directory)

        # Default extensions
        if file_extensions is None:
            if modality == 'image':
                file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            else:
                file_extensions = ['.wav', '.mp3', '.flac', '.ogg']

        # Find all files
        file_paths = []
        if recursive:
            for ext in file_extensions:
                file_paths.extend(dir_path.rglob(f'*{ext}'))
                file_paths.extend(dir_path.rglob(f'*{ext.upper()}'))
        else:
            for ext in file_extensions:
                file_paths.extend(dir_path.glob(f'*{ext}'))
                file_paths.extend(dir_path.glob(f'*{ext.upper()}'))

        file_paths = sorted(list(set([str(p) for p in file_paths])))

        print(f"Found {len(file_paths)} {modality} files in {directory}")

        # Predict
        if modality == 'image':
            return self.predict_images(file_paths, transform)
        else:
            raise NotImplementedError("Audio prediction not implemented yet")

    def save_results(
        self,
        results: List[Dict],
        output_path: Union[str, Path],
        format: str = 'json'
    ):
        """
        Save prediction results

        Args:
            results: List of prediction dictionaries
            output_path: Output file path
            format: 'json', 'csv', or 'txt'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

        elif format == 'csv':
            # Flatten nested dictionaries
            flat_results = []
            for r in results:
                if 'error' in r:
                    flat = {
                        'file_path': r['file_path'],
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'error': r['error']
                    }
                else:
                    flat = {
                        'file_path': r['file_path'],
                        'prediction': r['prediction'],
                        'confidence': r['confidence']
                    }
                    if 'class_probabilities' in r:
                        for cls, prob in r['class_probabilities'].items():
                            flat[f'prob_{cls}'] = prob
                flat_results.append(flat)

            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)

        elif format == 'txt':
            with open(output_path, 'w') as f:
                for r in results:
                    if 'error' in r:
                        f.write(f"{r['file_path']}: ERROR - {r['error']}\n")
                    else:
                        f.write(
                            f"{r['file_path']}: {r['prediction']} "
                            f"(confidence: {r['confidence']:.4f})\n"
                        )

        print(f"Results saved to {output_path}")

    def get_summary(self, results: List[Dict]) -> Dict:
        """Get summary statistics from results"""
        total = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        errors = total - successful

        if successful == 0:
            return {
                'total_files': total,
                'successful': 0,
                'errors': errors,
                'error_rate': 1.0
            }

        # Count predictions
        predictions = [r['prediction'] for r in results if 'error' not in r]
        stego_count = sum(1 for p in predictions if p == 'Stego')
        cover_count = successful - stego_count

        # Average confidence
        confidences = [r['confidence'] for r in results if 'error' not in r]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'total_files': total,
            'successful': successful,
            'errors': errors,
            'error_rate': errors / total if total > 0 else 0,
            'stego_detected': stego_count,
            'stego_percentage': stego_count / successful * 100 if successful > 0 else 0,
            'cover_images': cover_count,
            'cover_percentage': cover_count / successful * 100 if successful > 0 else 0,
            'average_confidence': float(avg_confidence)
        }

    def print_summary(self, results: List[Dict]):
        """Print summary of predictions"""
        summary = self.get_summary(results)

        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total files:            {summary['total_files']}")
        print(f"Successfully processed: {summary['successful']}")
        print(f"Errors:                 {summary['errors']}")
        print(f"Error rate:             {summary['error_rate']:.2%}")
        print()
        print(f"Stego detected:         {summary['stego_detected']} "
              f"({summary['stego_percentage']:.2f}%)")
        print(f"Cover images:           {summary['cover_images']} "
              f"({summary['cover_percentage']:.2f}%)")
        print(f"Average confidence:     {summary['average_confidence']:.4f}")
        print("="*60)
