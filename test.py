import torch
import os
import numpy as np
import wfdb
import scipy.signal
from model import PAFClassifier
from data_manager import get_loaders

import matplotlib
matplotlib.use('Agg') # Headless support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_test_record(data_dir, record_name, window_seconds=30, target_fs=128, target_channels=2):
    """Loads a single record, resamples, normalizes, and returns the LAST 30 seconds."""
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    
    # Get the signal (Channels, Samples)
    signal = record.p_signal.astype(np.float32).T 
    
    # 1. Handle Sampling Rate (Dynamic Resampling)
    if record.fs != target_fs:
        num_samples = int(signal.shape[1] * (target_fs / record.fs))
        signal = scipy.signal.resample(signal, num_samples, axis=1)

    # 2. Handle Channel Consistency
    if signal.shape[0] < target_channels:
        padding = np.zeros((target_channels - signal.shape[0], signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, padding])
    elif signal.shape[0] > target_channels:
        signal = signal[:target_channels, :]

    # 3. Slice the LAST window_samples (Best for predicting upcoming events)
    window_samples = window_seconds * target_fs
    if signal.shape[1] < window_samples:
        # Pad if the signal is somehow too short
        padding = window_samples - signal.shape[1]
        signal = np.pad(signal, ((0,0), (0, padding)), 'constant')
    else:
        # Grab the end of the recording
        start_idx = signal.shape[1] - window_samples
        signal = signal[:, start_idx:]
        
    # 4. Z-Score Normalization (MANDATORY for inference to match training)
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    signal = (signal - mean) / (std + 1e-8)
        
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0) # Add batch dim

def run_challenge_test(data_path='data/paf-prediction-challenge-database/', model_path='paf_resnet_best.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model and Load Weights
    model = PAFClassifier(in_channels=2, num_classes=2).to(device)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Train the model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # Identify Test Records (t01, t02, etc.)
    all_files = os.listdir(data_path)
    test_records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.startswith('t') and f.endswith('.hea')])))

    if not test_records:
        print("No 't' records found for challenge inference.")
        return

    print(f"Found {len(test_records)} test records. Starting inference...")
    results = {}

    with torch.no_grad(): 
        for name in test_records:
            try:
                input_tensor = load_test_record(data_path, name).to(device)
                
                output = model(input_tensor)
                
                _, predicted = torch.max(output, 1)
                prob = torch.softmax(output, dim=1)
                
                pred_label = predicted.item()
                confidence = prob[0][pred_label].item()

                results[name] = pred_label
                print(f"Record {name}: {'PRE-PAF' if pred_label == 1 else 'NORMAL'} (Conf: {confidence:.2f})")
            
            except Exception as e:
                print(f"Error processing {name}: {e}")

    print("\n--- Challenge Summary (Pairs) ---")
    for i in range(1, len(test_records), 2):
        if i < len(test_records):
            r1, r2 = test_records[i-1], test_records[i]
            print(f"Pair {r1}/{r2}: Prediction -> {results.get(r1)} / {results.get(r2)}")

# Changed default path to 'paf_resnet_best.pth'
def evaluate_on_val_set(model_path='paf_resnet_best.pth', batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_loader = get_loaders(batch_size=batch_size) 
    
    model = PAFClassifier(in_channels=2, num_classes=2).to(device)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Train the model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            
            outputs = model(signals)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    target_names = ['Normal/Distant (0)', 'Pre-PAF (1)']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: PAF Prediction')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_on_val_set()
    run_challenge_test()