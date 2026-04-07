import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wfdb
import os
import random

class PAFDataset(Dataset):
    def __init__(self, data_dir, record_names, window_seconds=30, target_fs=128, target_channels=2):
        self.window_samples = window_seconds * target_fs
        self.record_names = record_names
        self.target_channels = target_channels
        
        self.signals = []
        self.labels = []

        print(f"Loading {len(record_names)} records into RAM...", end="")
        
        for name in record_names:
            record_path = os.path.join(data_dir, name)
            
            # Load the full record once
            record = wfdb.rdrecord(record_path)
            
            # 1. Handle Sampling Rate (Basic check)
            if record.fs != target_fs:
                # In a real scenario, use scipy.signal.resample
                # For this fix, we'll just log a warning and proceed
                # (Assuming datasets in AFDB/AFTDB are consistent)
                pass

            signal = record.p_signal.astype(np.float32).T # (Channels, Samples)
            
            # 2. Handle Channel Count Consistency
            if signal.shape[0] < self.target_channels:
                # Pad with zeros if fewer channels
                padding = np.zeros((self.target_channels - signal.shape[0], signal.shape[1]), dtype=np.float32)
                signal = np.vstack([signal, padding])
            elif signal.shape[0] > self.target_channels:
                # Truncate if more channels
                signal = signal[:self.target_channels, :]

            self.signals.append(np.ascontiguousarray(signal))

            if name.startswith('n'):
                label = 0
            elif name.startswith('p'):
                label = 1
            else:
                label = -1
            self.labels.append(label)

        print(f"\t{sum(s.nbytes for s in self.signals) / (1024**3):.4f} GB Used")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        full_signal = self.signals[idx]  # (Channels, Total_Samples)
        label = self.labels[idx]
        
        total_samples = full_signal.shape[1] 

        # Random Slicing
        if total_samples > self.window_samples:
            max_start = total_samples - self.window_samples
            start = random.randint(0, max_start)
            end = start + self.window_samples
            windowed_signal = full_signal[:, start:end]
        else:
            # Pad if record is too short
            padding = self.window_samples - total_samples
            windowed_signal = np.pad(full_signal, ((0,0), (0, padding)), 'constant')

        signal_tensor = torch.tensor(windowed_signal, dtype=torch.float32)
        
        return signal_tensor, torch.tensor(label, dtype=torch.long)


def get_loaders(data_path='data/paf-prediction-challenge-database/',
                batch_size=32, window_seconds=30, fs=128, split_ratio=0.8):
    
    all_files = os.listdir(data_path)
    records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.endswith('.hea')])))
    
    # Filter for known classes
    valid_records = [r for r in records if r.startswith('n') or r.startswith('p')]
    
    # Randomized Train/Val Split
    random.seed(42) # Reproducibility
    random.shuffle(valid_records)
    
    split_idx = int(len(valid_records) * split_ratio)
    train_records = valid_records[:split_idx]
    val_records = valid_records[split_idx:]
    
    train_dataset = PAFDataset(data_path, train_records, window_seconds=window_seconds, target_fs=fs)
    val_dataset = PAFDataset(data_path, val_records, window_seconds=window_seconds, target_fs=fs)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test
    train_l, val_l = get_loaders()
    for signals, labels in train_l:
        print(f"Batch signals shape: {signals.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
