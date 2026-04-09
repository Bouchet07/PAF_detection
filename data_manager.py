import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wfdb
import os
import random
import re

class PAFDataset(Dataset):
    # ADDED 'mode' to __init__
    def __init__(self, data_dir, record_names, window_seconds=30, target_fs=128, target_channels=2, mode='train'):
        self.window_samples = window_seconds * target_fs
        self.record_names = record_names
        self.target_channels = target_channels
        self.mode = mode # 'train' or 'val'
        
        self.signals = []
        self.labels = []

        print(f"Loading {len(record_names)} records into RAM ({self.mode} mode)...", end="")
        
        for name in record_names:
            record_path = os.path.join(data_dir, name)
            record = wfdb.rdrecord(record_path)
            
            signal = record.p_signal.astype(np.float32).T 
            
            if signal.shape[0] < self.target_channels:
                padding = np.zeros((self.target_channels - signal.shape[0], signal.shape[1]), dtype=np.float32)
                signal = np.vstack([signal, padding])
            elif signal.shape[0] > self.target_channels:
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
        full_signal = self.signals[idx]  
        label = self.labels[idx]
        total_samples = full_signal.shape[1] 

        if total_samples > self.window_samples:
            max_start = total_samples - self.window_samples
            
            # Deterministic val, random train
            if self.mode == 'train':
                start = random.randint(0, max_start)
            else:
                # Always grab the exact middle of the signal for validation
                start = max_start // 2 
                
            end = start + self.window_samples
            windowed_signal = full_signal[:, start:end]
        else:
            padding = self.window_samples - total_samples
            windowed_signal = np.pad(full_signal, ((0,0), (0, padding)), 'constant')

        # Z-Score Normalization per channel
        # Add 1e-8 to avoid dividing by zero if a signal is totally flat
        mean = np.mean(windowed_signal, axis=1, keepdims=True)
        std = np.std(windowed_signal, axis=1, keepdims=True)
        windowed_signal = (windowed_signal - mean) / (std + 1e-8)

        signal_tensor = torch.tensor(windowed_signal, dtype=torch.float32)
        return signal_tensor, torch.tensor(label, dtype=torch.long)

def get_grouped_records(data_path='data/paf-prediction-challenge-database/', 
                        split_ratio=0.8, seed=42):
    all_files = os.listdir(data_path)
    records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.endswith('.hea')])))
    
    # Filter for learning set (n* and p*)
    valid_records = [r for r in records if r.startswith('n') or r.startswith('p')]

    # 1. Group records by Subject
    # Rules: 'p15', 'p15c', 'p16', 'p16c' all belong to one subject.
    # We identify subjects by (prefix + pair_index)
    subject_to_files = {}

    for r in valid_records:
        prefix = r[0] # 'n' or 'p'
        # Extract digits: e.g., 'p15c' -> 15
        num = int(re.search(r'\d+', r).group())
        
        # Determine the pair ID (1&2 are ID 0, 3&4 are ID 1, etc.)
        pair_id = (num - 1) // 2
        subject_id = f"{prefix}_{pair_id}"
        
        if subject_id not in subject_to_files:
            subject_to_files[subject_id] = []
        subject_to_files[subject_id].append(r)

    # 2. Shuffle the Subjects, not the files
    subjects = list(subject_to_files.keys())
    random.seed(seed)
    random.shuffle(subjects)

    # 3. Split the Subjects
    split_idx = int(len(subjects) * split_ratio)
    train_subjects = subjects[:split_idx]
    val_subjects = subjects[split_idx:]

    # 4. Map subjects back to their full list of files
    train_records = []
    for s in train_subjects:
        train_records.extend(subject_to_files[s])

    val_records = []
    for s in val_subjects:
        val_records.extend(subject_to_files[s])
    
    return train_records, val_records

def get_loaders(data_path='data/paf-prediction-challenge-database/',
                batch_size=32, window_seconds=30, fs=128, split_ratio=0.8, seed=42):
      
    train_records, val_records = get_grouped_records(data_path, split_ratio, seed)
    
    train_dataset = PAFDataset(data_path, train_records, window_seconds=window_seconds, target_fs=fs, mode='train')
    val_dataset = PAFDataset(data_path, val_records, window_seconds=window_seconds, target_fs=fs, mode='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test
    train_l, val_l = get_loaders()
    print(f'Train records: {train_l.dataset.record_names}')
    print(f'Val records: {val_l.dataset.record_names}')
    for signals, labels in train_l:
        print(f"Batch signals shape: {signals.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
