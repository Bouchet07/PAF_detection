import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

class PAFDataset(Dataset):
    """
    PyTorch Dataset for PAF imminence detection.
    Loads preprocessed 5-minute segments and extracts window slices.
    - Training mode: Random slice of window_seconds within the 5-minute block.
    - Eval/Val mode: Deterministic slice closest to the end of the block (closest to PAF onset).
    - Channel-wise Z-score normalization.
    """
    def __init__(
        self, 
        metadata: pd.DataFrame, 
        data_dir: str = 'processed_data', 
        window_seconds: int = 30, 
        target_fs: int = 128, 
        mode: str = 'train', 
        in_memory: bool = True,
        augment: bool = False,
        hrv_mean: np.ndarray = None,
        hrv_std: np.ndarray = None
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.data_dir = data_dir
        self.window_samples = window_seconds * target_fs
        self.mode = mode
        self.in_memory = in_memory
        self.augment = augment
        self.target_fs = target_fs
        
        # Load and normalize HRV features
        self.hrv_cols = ['mean_rr', 'std_rr', 'rmssd', 'pnn50', 'mean_hr', 'std_hr', 'lf', 'hf', 'lf_hf_ratio']
        raw_hrv = self.metadata[self.hrv_cols].values.astype(np.float32)
        
        if self.mode == 'train':
            self.hrv_mean = np.mean(raw_hrv, axis=0)
            self.hrv_std = np.std(raw_hrv, axis=0) + 1e-8
        else:
            self.hrv_mean = hrv_mean if hrv_mean is not None else np.zeros(len(self.hrv_cols), dtype=np.float32)
            self.hrv_std = hrv_std if hrv_std is not None else np.ones(len(self.hrv_cols), dtype=np.float32)
            
        self.hrv_features = (raw_hrv - self.hrv_mean) / self.hrv_std
        
        self.signals = {}
        if self.in_memory:
            print(f"Loading {len(metadata)} segments into RAM for {mode}...")
            for _, row in metadata.iterrows():
                file_path = os.path.join(self.data_dir, row['filename'])
                if os.path.exists(file_path):
                    self.signals[row['filename']] = np.load(file_path).astype(np.float32)
                else:
                    print(f"Warning: File {file_path} not found.")

    def __len__(self):
        return len(self.metadata)

    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        t = np.arange(signal.shape[1]) / self.target_fs
        freq = np.random.uniform(0.1, 0.5)
        amp = np.random.uniform(0.05, 0.2)
        drift = amp * np.sin(2 * np.pi * freq * t)
        return signal + drift

    def _add_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, np.random.uniform(0.01, 0.05), signal.shape).astype(np.float32)
        return signal + noise

    def _random_scale(self, signal: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(0.85, 1.15)
        return signal * scale

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        label = row['label']
        
        if self.in_memory and filename in self.signals:
            full_signal = self.signals[filename]
        else:
            file_path = os.path.join(self.data_dir, filename)
            full_signal = np.load(file_path).astype(np.float32)
            
        total_samples = full_signal.shape[1]
        
        # Extract a window of size self.window_samples
        if total_samples > self.window_samples:
            max_start = total_samples - self.window_samples
            if self.mode == 'train':
                start = random.randint(0, max_start)
            else:
                start = max_start
            
            end = start + self.window_samples
            windowed_signal = full_signal[:, start:end]
        else:
            # Pad with constant if the signal is shorter than window_samples
            padding = self.window_samples - total_samples
            windowed_signal = np.pad(full_signal, ((0,0), (0, padding)), 'constant')

        # Apply data augmentations if training and augment is True
        if self.mode == 'train' and self.augment:
            windowed_signal = windowed_signal.copy()
            if random.random() < 0.5:
                windowed_signal = self._add_baseline_wander(windowed_signal)
            if random.random() < 0.5:
                windowed_signal = self._add_gaussian_noise(windowed_signal)
            if random.random() < 0.5:
                windowed_signal = self._random_scale(windowed_signal)

        # Channel-wise Z-Score Normalization
        mean = np.mean(windowed_signal, axis=1, keepdims=True)
        std = np.std(windowed_signal, axis=1, keepdims=True)
        windowed_signal = (windowed_signal - mean) / (std + 1e-8)

        hrv_tensor = torch.tensor(self.hrv_features[idx], dtype=torch.float32)

        return (
            torch.tensor(windowed_signal, dtype=torch.float32), 
            hrv_tensor,
            torch.tensor(label, dtype=torch.long)
        )

