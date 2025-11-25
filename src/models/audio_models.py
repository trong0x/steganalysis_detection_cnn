"""
Audio-specific models for steganalysis (spectrogram-based CNNs)
"""

import torch
import torch.nn as nn


class AudioStegCNN(nn.Module):
    """CNN for audio steganalysis using spectrograms"""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super(AudioStegCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class RNNAudioSteg(nn.Module):
    """RNN-based model for sequential audio analysis"""

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True
    ):
        super(RNNAudioSteg, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time)
        # Reshape for LSTM: (batch, time, features)
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Classification
        output = self.classifier(context)
        return output


class HybridAudioSteg(nn.Module):
    """Hybrid CNN-RNN model for audio steganalysis"""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super(HybridAudioSteg, self).__init__()

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # RNN for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        batch_size = x.size(0)
        x = self.cnn(x)

        # Reshape for LSTM: (batch, time, features)
        x = x.mean(dim=2)  # Average over frequency
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Use last hidden state
        x = lstm_out[:, -1, :]

        # Classification
        x = self.classifier(x)
        return x


class Wav2VecSteg(nn.Module):
    """Simplified wav2vec-inspired model for audio steganalysis"""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(Wav2VecSteg, self).__init__()

        # Convolutional feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature encoding
        x = self.feature_encoder(x)

        # Reshape for transformer: (batch, seq, features)
        x = x.permute(0, 2, 1)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)
        return x
