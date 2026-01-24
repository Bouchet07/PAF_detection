import torch
from torch.utils.data import DataLoader, Dataset

import wfdb
import os
import random

class PAFDataset(Dataset):
    def __init__(self, data_dir, record_names, window_seconds=30, fs=128):
        self.data_dir = data_dir
        self.record_names = record_names
        self.window_samples = window_seconds * fs # 30 * 128 = 3840

    def __len__(self):
        return len(self.record_names)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_dir, self.record_names[idx])
        
        # 1. Load the full 30-minute record
        record = wfdb.rdrecord(record_path)
        full_signal = record.p_signal  # (samples, channels)
        total_samples = full_signal.shape[0]

        # 2. Pick a random start point for the 30s window
        # Ensure we don't start too late and run out of signal
        max_start = total_samples - self.window_samples
        start = random.randint(0, max_start)
        end = start + self.window_samples
        
        windowed_signal = full_signal[start:end, :]
        
        # 3. Format for PyTorch: (Channels, Length)
        signal_tensor = torch.tensor(windowed_signal, dtype=torch.float32).t()
        
        # 4. Labeling Logic
        name = self.record_names[idx]
        if name.startswith('n'):
            label = 0
        elif name.startswith('p'):
            label = 1
        else:
            label = -1 
            
        return signal_tensor, torch.tensor(label, dtype=torch.long)
    

def get_loaders(data_path='data/paf-prediction-challenge-database/', batch_size=32, window_seconds=30, fs=128):
    """Factory function to create loaders safely."""
    all_files = os.listdir(data_path)
    records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.endswith('.hea')])))
    train_records = [r for r in records if r.startswith('n') or r.startswith('p')]
    
    dataset = PAFDataset(data_path, train_records, window_seconds=window_seconds, fs=fs)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True # Bonus: makes transferring to GPU faster
    )
    return loader


if __name__ == "__main__":
    # Simple test to verify the loader works
    loader = get_loaders()
    for signals, labels in loader:
        print(f"Batch signals shape: {signals.shape}")  # Expect (batch_size, 2, 3840)
        print(f"Batch labels shape: {labels.shape}")    # Expect (batch_size,)
        break