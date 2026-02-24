import torch
import os
import numpy as np
import wfdb
from model import PAFClassifier # Assuming your model is in model.py
from data_manager import get_loaders

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_test_record(data_dir, record_name, window_seconds=30, fs=128):
    """Loads a single record and returns a tensor of the first 30 seconds."""
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    
    # Get the signal (Channels, Samples)
    signal = record.p_signal.astype(np.float32).T 
    
    # We take the first 'window_samples' to keep it consistent
    window_samples = window_seconds * fs
    if signal.shape[1] < window_samples:
        # Pad if the signal is somehow too short
        padding = window_samples - signal.shape[1]
        signal = np.pad(signal, ((0,0), (0, padding)), 'constant')
    else:
        signal = signal[:, :window_samples]
        
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0) # Add batch dim

def run_test(data_path='data/paf-prediction-challenge-database/', model_path='paf_resnet.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model and Load Weights
    model = PAFClassifier(in_channels=2, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # CRITICAL: Set to evaluation mode (turns off Dropout/BatchNorm)

    # 2. Identify Test Records (t01, t02, etc.)
    all_files = os.listdir(data_path)
    test_records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.startswith('t') and f.endswith('.hea')])))

    print(f"Found {len(test_records)} test records. Starting inference...")

    results = {}

    with torch.no_grad(): # Disable gradient calculation for speed and memory
        for name in test_records:
            try:
                # Load and move to device
                input_tensor = load_test_record(data_path, name).to(device)
                
                # Forward Pass
                output = model(input_tensor)
                
                # Get Prediction (0 = Normal/Distant, 1 = Pre-PAF)
                _, predicted = torch.max(output, 1)
                prob = torch.softmax(output, dim=1)
                
                pred_label = predicted.item()
                confidence = prob[0][pred_label].item()

                results[name] = pred_label
                print(f"Record {name}: {'PRE-PAF' if pred_label == 1 else 'NORMAL'} (Conf: {confidence:.2f})")
            
            except Exception as e:
                print(f"Error processing {name}: {e}")

    # 3. Group by Pairs (t01/t02, t03/t04) as per Challenge rules
    print("\n--- Challenge Summary (Pairs) ---")
    for i in range(1, len(test_records), 2):
        r1, r2 = test_records[i-1], test_records[i]
        print(f"Pair {r1}/{r2}: Prediction -> {results.get(r1)} / {results.get(r2)}")

def evaluate_model(model_path='paf_resnet.pth', batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    # Note: In a real scenario, you'd pass a specific 'val_records' list here
    # to ensure you aren't evaluating on data the model already saw.
    val_loader = get_loaders(batch_size=batch_size) 
    
    # 2. Load Model
    model = PAFClassifier(in_channels=2, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation...")
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            
            outputs = model(signals)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Scikit-Learn Metrics
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    target_names = ['Normal/Distant (0)', 'Pre-PAF (1)']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 4. Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: PAF Prediction')
    plt.show()

if __name__ == "__main__":
    evaluate_model()