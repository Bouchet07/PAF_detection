import pandas as pd
import random
from typing import Set

def verify_data_leakage(metadata_path: str = 'metadata.csv', split_ratio: float = 0.8, seed: int = 42) -> bool:
    """
    Checks if there is any data leakage (subject or file overlap) between the train and val splits.
    
    Returns:
        True if the split is safe (no leakage), False otherwise.
    """
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Error: {metadata_path} not found. Cannot verify leakage.")
        return False
        
    # Identify unique subjects
    subjects = df['subject'].unique()
    subjects_list = list(subjects)
    
    random.seed(seed)
    random.shuffle(subjects_list)
    
    split_idx = int(len(subjects_list) * split_ratio)
    train_subjects = set(subjects_list[:split_idx])
    val_subjects = set(subjects_list[split_idx:])
    
    # Check for subject intersection
    subject_intersection = train_subjects.intersection(val_subjects)
    
    print("\n--- Leakage Verification ---")
    print(f"Total Unique Subjects: {len(subjects)}")
    print(f"Train Subjects: {len(train_subjects)}")
    print(f"Val Subjects: {len(val_subjects)}")
    
    no_leakage = True
    
    if not subject_intersection:
        print("[SUCCESS] No subject leakage detected (Train and Val subjects are completely disjoint).")
    else:
        print(f"[FAILURE] Subject leakage detected! Intersection: {subject_intersection}")
        no_leakage = False
        
    train_df = df[df['subject'].isin(train_subjects)]
    val_df = df[df['subject'].isin(val_subjects)]
    
    print(f"Train Records: {len(train_df)}")
    print(f"Val Records: {len(val_df)}")
    
    # Check if any individual filename appears in both (extra safety)
    file_intersection = set(train_df['filename']).intersection(set(val_df['filename']))
    if not file_intersection:
        print("[SUCCESS] No file path leakage detected.")
    else:
        print(f"[FAILURE] {len(file_intersection)} files appear in both train and val sets!")
        no_leakage = False
        
    return no_leakage

if __name__ == "__main__":
    verify_data_leakage()
