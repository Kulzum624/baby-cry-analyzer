# ===========================
# dataset.py - Baby Cry Dataset with Audio Processing
# ===========================
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

class BabyCryDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data = []
        self.labels = []
        self.classes = ['hungry', 'belly_pain', 'burping', 'discomfort', 'tired']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.augment = augment

        # Audio parameters
        self.target_sample_rate = 16000  # Standard sample rate for audio classification
        self.target_length = 16000 * 5  # 5 seconds of audio

        # Load data paths
        for label in self.classes:
            folder = os.path.join(data_dir, label)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith('.wav'):
                    self.data.append(os.path.join(folder, file))
                    self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.data)

    def _random_gain(self, waveform):
        """Apply random gain between 0.8 and 1.2"""
        gain = torch.FloatTensor(1).uniform_(0.8, 1.2)
        return waveform * gain

    def _add_noise(self, waveform, noise_level=0.005):
        """Add random noise to waveform"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def _time_shift(self, waveform, shift_limit=0.2):
        """Shift waveform in time"""
        shift = int(random.uniform(-shift_limit, shift_limit) * waveform.shape[1])
        return torch.roll(waveform, shift, dims=1)

    def pad_or_truncate(self, waveform):
        if waveform.size(1) > self.target_length:
            # Random crop if too long
            start = random.randint(0, waveform.size(1) - self.target_length)
            waveform = waveform[:, start:start + self.target_length]
        elif waveform.size(1) < self.target_length:
            # Pad with reflection padding
            padding = self.target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding), mode='reflect')
        return waveform

    def _augment_waveform(self, waveform):
        """Apply audio augmentation to waveform"""
        if random.random() < 0.5:
            # Random gain
            waveform = self._random_gain(waveform)
            
        if random.random() < 0.5:
            # Add random noise
            waveform = self._add_noise(waveform)
            
        if random.random() < 0.5:
            # Time shift
            waveform = self._time_shift(waveform)
            
        return waveform

    def _normalize_waveform(self, waveform):
        """Normalize waveform to have zero mean and unit variance"""
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-8)
        return waveform

    def __getitem__(self, idx):
        try:
            # Load audio
            waveform, sr = torchaudio.load(self.data[idx])
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                waveform = resampler(waveform)

            # Ensure fixed length first
            waveform = self.pad_or_truncate(waveform)

            # Apply augmentation to waveform if enabled
            if self.augment:
                waveform = self._augment_waveform(waveform)
            
            # Normalize waveform
            waveform = self._normalize_waveform(waveform)

            return waveform.squeeze(0), self.labels[idx]
            
        except Exception as e:
            print(f"Error processing {self.data[idx]}: {str(e)}")
            # Return a zero tensor and the label if there's an error
            return torch.zeros(self.target_length), self.labels[idx]
