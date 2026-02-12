import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wfdb
import os
import random

class PAFDataset(Dataset):
    def __init__(self, data_dir, record_names, window_seconds=30, fs=128):
        self.window_samples = window_seconds * fs
        self.record_names = record_names
        
        self.signals = []
        self.labels = []

        print(f"Loading {len(record_names)} records into RAM...", end="")
        
        for name in record_names:
            record_path = os.path.join(data_dir, name)
            
            #Load the full record once
            record = wfdb.rdrecord(record_path)

            signal = record.p_signal.astype(np.float32).T 
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
        max_start = total_samples - self.window_samples
        
        start = random.randint(0, max_start)
        end = start + self.window_samples
        windowed_signal = full_signal[:, start:end] 

        signal_tensor = torch.tensor(windowed_signal, dtype=torch.float32)
        
        return signal_tensor, torch.tensor(label, dtype=torch.long)


def get_loaders(data_path='data/paf-prediction-challenge-database/',
                batch_size=32, window_seconds=30, fs=128):
    
    all_files = os.listdir(data_path)
    records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.endswith('.hea')])))
    
    train_records = [r for r in records if r.startswith('n') or r.startswith('p')]
    
    dataset = PAFDataset(data_path, train_records, window_seconds=window_seconds, fs=fs)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    return loader

if __name__ == "__main__":
    # Test
    loader = get_loaders()
    for signals, labels in loader:
        print(f"Batch signals shape: {signals.shape}") # (Batch, Channels, Samples)
        print(f"Batch labels shape: {labels.shape}")
        break