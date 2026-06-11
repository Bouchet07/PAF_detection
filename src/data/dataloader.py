import os
import random
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple
from src.data.dataset import PAFDataset

def get_loaders(
    metadata_path: str = 'metadata.csv', 
    data_dir: str = 'processed_data', 
    batch_size: int = 32, 
    window_seconds: int = 30, 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    in_memory: bool = True,
    augment: bool = False,
    use_sampler: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits data by subject into Train, Validation, and Test sets and builds PyTorch DataLoaders.
    
    Args:
        metadata_path: Path to the metadata CSV file.
        data_dir: Directory containing preprocessed .npy files.
        batch_size: Batch size for DataLoader.
        window_seconds: Input window duration in seconds.
        train_ratio: Ratio of subjects allocated to training.
        val_ratio: Ratio of subjects allocated to validation (dev).
        seed: Random seed for split reproducibility.
        in_memory: If True, loads all segments into RAM.
        augment: If True, applies random on-the-fly augmentations to train dataset.
        use_sampler: If True, uses WeightedRandomSampler to balance batch classes.
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. Please run the preprocessing pipeline first."
        )
        
    df = pd.read_csv(metadata_path)
    
    # Extract unique subjects for subject-based disjoint split
    subjects = df['subject'].unique()
    
    # Shuffle subjects deterministically
    random.seed(seed)
    subjects_list = list(subjects)
    random.shuffle(subjects_list)
    
    # Compute split bounds
    total_subjects = len(subjects_list)
    split1 = int(total_subjects * train_ratio)
    split2 = int(total_subjects * (train_ratio + val_ratio))
    
    train_subjects = set(subjects_list[:split1])
    val_subjects = set(subjects_list[split1:split2])
    test_subjects = set(subjects_list[split2:])
    
    # Filter records
    train_df = df[df['subject'].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df['subject'].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_subjects)].reset_index(drop=True)
    
    print(f"--- Data Loader Setup ---")
    print(f"Total Subjects: {total_subjects} (Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)})")
    print(f"Total Segments: {len(df)} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    
    # Print label distribution
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        counts = split_df['label'].value_counts()
        dist = split_df['label'].value_counts(normalize=True)
        print(f"{name} Label Distribution:")
        for lbl in [0, 1]:
            c = counts.get(lbl, 0)
            d = dist.get(lbl, 0.0) * 100
            print(f"  Label {lbl} (Normal if 0, Pre-PAF if 1): {c} segments ({d:.2f}%)")
            
    # Instantiate Datasets
    train_dataset = PAFDataset(
        metadata=train_df, 
        data_dir=data_dir, 
        window_seconds=window_seconds, 
        mode='train', 
        in_memory=in_memory,
        augment=augment
    )
    
    # Extract training set's HRV statistics to normalize validation and test sets without leakage
    hrv_mean = train_dataset.hrv_mean
    hrv_std = train_dataset.hrv_std
    
    val_dataset = PAFDataset(
        metadata=val_df, 
        data_dir=data_dir, 
        window_seconds=window_seconds, 
        mode='val', 
        in_memory=in_memory,
        hrv_mean=hrv_mean,
        hrv_std=hrv_std
    )
    test_dataset = PAFDataset(
        metadata=test_df, 
        data_dir=data_dir, 
        window_seconds=window_seconds, 
        mode='val',  # test set acts as deterministic validation/inference
        in_memory=in_memory,
        hrv_mean=hrv_mean,
        hrv_std=hrv_std
    )
    
    # Build DataLoaders
    if use_sampler:
        import numpy as np
        import torch
        train_labels = train_df['label'].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader


def get_kfold_loaders(
    metadata_path: str = 'metadata.csv', 
    data_dir: str = 'processed_data', 
    batch_size: int = 32, 
    window_seconds: int = 30, 
    k_fold: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    in_memory: bool = True,
    augment: bool = False,
    use_sampler: bool = False
) -> Tuple[list, DataLoader, list]:
    """
    Splits cv_subjects into k disjoint Group Folds and returns a list of (train_loader, val_loader)
    for each fold, along with a consistent hidden test loader.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
    df = pd.read_csv(metadata_path)
    
    # Extract unique subjects
    subjects = df['subject'].unique()
    
    # Shuffle subjects deterministically
    random.seed(seed)
    subjects_list = list(subjects)
    random.shuffle(subjects_list)
    
    # Split test set (same test set as before to keep a consistent holdout)
    total_subjects = len(subjects_list)
    split_idx = int(total_subjects * (train_ratio + val_ratio))
    cv_subjects = subjects_list[:split_idx]
    test_subjects = subjects_list[split_idx:]
    
    test_df = df[df['subject'].isin(test_subjects)].reset_index(drop=True)
    
    # Group K-Fold on cv_subjects
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    
    folds_loaders = []
    
    for fold, (train_subj_idx, val_subj_idx) in enumerate(kf.split(cv_subjects)):
        fold_train_subjs = set([cv_subjects[i] for i in train_subj_idx])
        fold_val_subjs = set([cv_subjects[i] for i in val_subj_idx])
        
        train_df = df[df['subject'].isin(fold_train_subjs)].reset_index(drop=True)
        val_df = df[df['subject'].isin(fold_val_subjs)].reset_index(drop=True)
        
        # Instantiate Datasets
        train_dataset = PAFDataset(
            metadata=train_df, 
            data_dir=data_dir, 
            window_seconds=window_seconds, 
            mode='train', 
            in_memory=in_memory,
            augment=augment
        )
        
        hrv_mean = train_dataset.hrv_mean
        hrv_std = train_dataset.hrv_std
        
        val_dataset = PAFDataset(
            metadata=val_df, 
            data_dir=data_dir, 
            window_seconds=window_seconds, 
            mode='val', 
            in_memory=in_memory,
            hrv_mean=hrv_mean,
            hrv_std=hrv_std
        )
        
        # Build DataLoaders
        if use_sampler:
            import numpy as np
            import torch
            train_labels = train_df['label'].values
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[train_labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        folds_loaders.append((train_loader, val_loader))
        
    # Baseline for test set normalization (same as standard loaders)
    cv_df = df[df['subject'].isin(cv_subjects)].reset_index(drop=True)
    baseline_train = PAFDataset(
        metadata=cv_df, 
        data_dir=data_dir, 
        window_seconds=window_seconds, 
        mode='train', 
        in_memory=False
    )
    test_dataset = PAFDataset(
        metadata=test_df,
        data_dir=data_dir,
        window_seconds=window_seconds,
        mode='val',
        in_memory=in_memory,
        hrv_mean=baseline_train.hrv_mean,
        hrv_std=baseline_train.hrv_std
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return folds_loaders, test_loader, test_subjects



