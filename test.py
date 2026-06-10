import os
import json
import torch
import numpy as np
import pandas as pd
import wfdb
import scipy.signal
from typing import Tuple
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.models import PAFClassifier, CNNTransformerPAFClassifier, SEResNetPAFClassifier
from src.data.dataloader import get_loaders
from src.utils.metrics import plot_confusion_matrix
from src.data.preprocess import compute_hrv_features

def load_test_record(
    data_dir: str, 
    record_name: str, 
    window_seconds: int = 30, 
    target_fs: int = 128, 
    target_channels: int = 2,
    hrv_mean: np.ndarray = None,
    hrv_std: np.ndarray = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a single test record, resamples, standardizes channels, normalizes,
    and returns the LAST window_seconds as a tensor, along with its normalized HRV features.
    """
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    
    # 1. Load QRS peaks from automated detector .qrs file
    try:
        ann = wfdb.rdann(record_path, 'qrs')
        r_peaks = ann.sample
    except Exception:
        r_peaks = []
        
    # Get the signal (Channels, Samples)
    signal = record.p_signal.astype(np.float32).T 
    
    # 2. Resample to target sampling rate
    if record.fs != target_fs:
        num_samples = int(signal.shape[1] * (target_fs / record.fs))
        signal = scipy.signal.resample(signal, num_samples, axis=1)
        scale_factor = target_fs / record.fs
    else:
        scale_factor = 1.0

    # 3. Channel Consistency
    if signal.shape[0] < target_channels:
        padding = np.zeros((target_channels - signal.shape[0], signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, padding])
    elif signal.shape[0] > target_channels:
        signal = signal[:target_channels, :]

    # 4. Slice the last window_samples
    window_samples = window_seconds * target_fs
    if signal.shape[1] < window_samples:
        padding = window_samples - signal.shape[1]
        signal = np.pad(signal, ((0,0), (0, padding)), 'constant')
        start_idx_orig = 0
    else:
        start_idx = signal.shape[1] - window_samples
        signal = signal[:, start_idx:]
        start_idx_orig = int(start_idx / scale_factor)
        
    # 5. Z-Score Normalization of signal
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    signal = (signal - mean) / (std + 1e-8)
    
    # 6. Extract and normalize HRV features
    end_idx_orig = start_idx_orig + int(window_samples / scale_factor)
    peaks_in_window = [p - start_idx_orig for p in r_peaks if start_idx_orig <= p < end_idx_orig]
    rescaled_peaks = [int(p * scale_factor) for p in peaks_in_window]
    
    hrv = compute_hrv_features(rescaled_peaks, target_fs)
    
    # Pack HRV features
    hrv_cols = ['mean_rr', 'std_rr', 'rmssd', 'pnn50', 'mean_hr', 'std_hr', 'lf', 'hf', 'lf_hf_ratio']
    hrv_raw = np.array([hrv[c] for c in hrv_cols], dtype=np.float32)
    
    if hrv_mean is not None and hrv_std is not None:
        hrv_norm = (hrv_raw - hrv_mean) / hrv_std
    else:
        hrv_norm = hrv_raw
        
    return (
        torch.tensor(signal, dtype=torch.float32).unsqueeze(0),
        torch.tensor(hrv_norm, dtype=torch.float32).unsqueeze(0)
    )

def get_latest_run_dir(results_dir: str = 'results') -> str:
    """Returns the most recently created run directory under results."""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory '{results_dir}' does not exist. Please train a model first.")
    runs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not runs:
        raise FileNotFoundError(f"No run directories found in '{results_dir}'.")
    return max(runs, key=os.path.getctime)

def load_model_from_run(run_dir: str, device: torch.device):
    """Loads configuration and weights from a run folder and instantiates the correct architecture."""
    config_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "best_model.pth")
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing config.json or best_model.pth in {run_dir}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model_type = config.get("model_type", "resnet1d").lower()
    window_seconds = config.get("window_seconds", 10)
    use_hrv = config.get("use_hrv", False)
    hrv_dim = 9 if use_hrv else 0
    
    # Instantiate correct architecture
    if model_type == 'resnet1d':
        model = PAFClassifier(in_channels=2, num_classes=2, hrv_dim=hrv_dim).to(device)
    elif model_type == 'transformer':
        model = CNNTransformerPAFClassifier(in_channels=2, num_classes=2, hrv_dim=hrv_dim).to(device)
    elif model_type == 'senet':
        model = SEResNetPAFClassifier(in_channels=2, num_classes=2, hrv_dim=hrv_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded {model_type.upper()} model from: {run_dir} (use_hrv: {use_hrv})")
    return model, window_seconds, use_hrv, config.get("hrv_mean"), config.get("hrv_std")

def run_challenge_test(run_dir: str, data_path: str = 'data/paf-prediction-challenge-database/'):
    """Runs challenge predictions using the model from a run directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_seconds, use_hrv, hrv_mean, hrv_std = load_model_from_run(run_dir, device)
    
    if hrv_mean is not None:
        hrv_mean = np.array(hrv_mean, dtype=np.float32)
    if hrv_std is not None:
        hrv_std = np.array(hrv_std, dtype=np.float32)
    
    if not os.path.exists(data_path):
        print(f"Challenge database path {data_path} not found. Skipping challenge test.")
        return
 
    # Find test records starting with 't'
    all_files = os.listdir(data_path)
    test_records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.startswith('t') and f.endswith('.hea')])))

    if not test_records:
        print("No challenge test 't' records found.")
        return

    print(f"Running challenge inference on {len(test_records)} records...")
    results = {}

    with torch.no_grad(): 
        for name in test_records:
            try:
                input_tensor, hrv_tensor = load_test_record(
                    data_path, name, window_seconds=window_seconds,
                    hrv_mean=hrv_mean, hrv_std=hrv_std
                )
                input_tensor = input_tensor.to(device)
                hrv_tensor = hrv_tensor.to(device)
                
                if use_hrv:
                    output = model(input_tensor, hrv_tensor)
                else:
                    output = model(input_tensor)
                
                _, predicted = torch.max(output, 1)
                prob = torch.softmax(output, dim=1)
                
                pred_label = predicted.item()
                confidence = prob[0][pred_label].item()

                results[name] = pred_label
                print(f"Record {name}: {'PRE-PAF' if pred_label == 1 else 'NORMAL'} (Conf: {confidence:.2f})")
            
            except Exception as e:
                print(f"Error processing challenge record {name}: {e}")

    print("\n--- Challenge Evaluation Summary (Subject Pairs) ---")
    for i in range(1, len(test_records), 2):
        if i < len(test_records):
            r1, r2 = test_records[i-1], test_records[i]
            print(f"Pair {r1}/{r2} Predictions: {results.get(r1, 'N/A')} / {results.get(r2, 'N/A')}")
            
    # Save predictions in the run folder
    predictions_path = os.path.join(run_dir, "challenge_predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved challenge predictions to: {predictions_path}")

def evaluate_on_test_set(run_dir: str, metadata_path: str = 'metadata.csv', data_dir: str = 'processed_data', batch_size: int = 32):
    """Evaluates the trained model run on its specific hidden Test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_seconds, use_hrv, hrv_mean, hrv_std = load_model_from_run(run_dir, device)
    
    if hrv_mean is not None:
        hrv_mean = np.array(hrv_mean, dtype=np.float32)
    if hrv_std is not None:
        hrv_std = np.array(hrv_std, dtype=np.float32)
        
    # Read config to get test_subjects
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    test_subjects = config.get("test_subjects")
    if test_subjects is None:
        raise ValueError(f"No test_subjects list found in {config_path}. Cannot perform unbiased evaluation.")
        
    # Load metadata and filter by test_subjects
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
        
    df = pd.read_csv(metadata_path)
    test_df = df[df['subject'].isin(test_subjects)].reset_index(drop=True)
    
    print(f"Loaded {len(test_df)} test segments for {len(test_subjects)} subjects.")
    
    # Build Test Dataset
    from src.data.dataset import PAFDataset
    
    test_dataset = PAFDataset(
        metadata=test_df,
        data_dir=data_dir,
        window_seconds=window_seconds,
        mode='val',  # test set acts as deterministic validation/inference
        in_memory=True,
        hrv_mean=hrv_mean,
        hrv_std=hrv_std
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []

    print("Running evaluation on hidden test set...")
    with torch.no_grad():
        for signals, hrvs, labels in test_loader:
            signals = signals.to(device)
            hrvs = hrvs.to(device)
            
            if use_hrv:
                outputs = model(signals, hrvs)
            else:
                outputs = model(signals)
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*40)
    print("UNBIASED TEST CLASSIFICATION REPORT")
    print("="*40)
    target_names = ['Normal/Distant (0)', 'Pre-PAF (1)']
    report_text = classification_report(all_labels, all_preds, target_names=target_names)
    print(report_text)

    # Save test reports & plots inside the run folder
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, save_path=os.path.join(run_dir, 'test_confusion_matrix.png'), title=f'Test Confusion Matrix: {os.path.basename(run_dir)}')
    
    report_path = os.path.join(run_dir, 'test_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved unbiased test evaluation files to: {run_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained PAF prediction model on the test set and run challenge prediction.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to the training run directory (e.g. results/resnet1d_win10_lr0.001_...). If not specified, defaults to the latest run.")
    parser.add_argument("--metadata_path", type=str, default="metadata.csv",
                        help="Path to preprocessed metadata.csv")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                        help="Directory containing preprocessed NumPy segment files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args()
    
    try:
        run_dir = args.run_dir
        if run_dir is None:
            run_dir = get_latest_run_dir()
        evaluate_on_test_set(run_dir, metadata_path=args.metadata_path, data_dir=args.data_dir, batch_size=args.batch_size)
        run_challenge_test(run_dir)
    except Exception as e:
        print(f"Error: {e}")