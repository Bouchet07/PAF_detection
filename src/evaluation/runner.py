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

from src.models import instantiate_model
from src.utils.metrics import plot_confusion_matrix, plot_eval_curves
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

def load_model_from_run(run_dir: str, device: torch.device):
    """Loads configuration and weights from a run folder and instantiates the correct architecture."""
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model_type = config.get("model_type", "resnet1d").lower()
    window_seconds = config.get("window_seconds", 10)
    use_hrv = config.get("use_hrv", False)
    k_fold = config.get("k_fold", 1)
    hrv_dim = 9 if use_hrv else 0
    
    # Look in structured checkpoints folder first, fallback to fallback root
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        checkpoints_dir = run_dir
        
    if k_fold > 1:
        models = []
        for fold_idx in range(k_fold):
            model_path = os.path.join(checkpoints_dir, f"best_model_fold_{fold_idx}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing weights for fold {fold_idx} at {model_path}")
            model = instantiate_model(model_type, hrv_dim, device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
        print(f"Loaded ensemble of {k_fold} {model_type.upper()} models from: {run_dir} (use_hrv: {use_hrv})")
        return models, window_seconds, use_hrv, config.get("hrv_mean"), config.get("hrv_std"), k_fold
    else:
        model_path = os.path.join(checkpoints_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing best_model.pth in {run_dir}")
        model = instantiate_model(model_type, hrv_dim, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded {model_type.upper()} model from: {run_dir} (use_hrv: {use_hrv})")
        return [model], window_seconds, use_hrv, config.get("hrv_mean"), config.get("hrv_std"), k_fold

def evaluate_on_test_set(
    run_dir: str, 
    metadata_path: str = 'metadata.csv', 
    data_dir: str = 'processed_data', 
    batch_size: int = 32
):
    """Evaluates the trained model run on its specific hidden Test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, window_seconds, use_hrv, hrv_mean, hrv_std, k_fold = load_model_from_run(run_dir, device)
    
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
    all_probs = []

    print("Running evaluation on hidden test set...")
    with torch.no_grad():
        for signals, hrvs, labels in test_loader:
            signals = signals.to(device)
            hrvs = hrvs.to(device)
            
            # Compute probabilities for each model and average them
            batch_probs = []
            for model in models:
                if use_hrv:
                    outputs = model(signals, hrvs)
                else:
                    outputs = model(signals)
                probs = torch.softmax(outputs, dim=1)[:, 1] # shape (batch_size,)
                batch_probs.append(probs.cpu().numpy())
            
            avg_probs = np.mean(batch_probs, axis=0) # shape (batch_size,)
            preds = (avg_probs >= 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(avg_probs)
            all_labels.extend(labels.numpy())

    print("\n" + "="*40)
    print("UNBIASED TEST CLASSIFICATION REPORT")
    print("="*40)
    target_names = ['Normal/Distant (0)', 'Pre-PAF (1)']
    report_text = classification_report(all_labels, all_preds, target_names=target_names)
    print(report_text)

    # Set up structured subfolders
    plots_test_dir = os.path.join(run_dir, "plots", "test")
    reports_test_dir = os.path.join(run_dir, "reports", "test")
    os.makedirs(plots_test_dir, exist_ok=True)
    os.makedirs(reports_test_dir, exist_ok=True)

    # Save test reports & plots inside the structured folders
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, save_path=os.path.join(plots_test_dir, 'test_confusion_matrix.png'), title=f'Test Confusion Matrix: {os.path.basename(run_dir)}')
    
    # Plot and save ROC and PR curves
    plot_eval_curves(
        y_true=np.array(all_labels),
        y_probs=np.array(all_probs),
        save_path=os.path.join(plots_test_dir, 'test_evaluation_curves.png'),
        title_prefix=f'Test Set ({os.path.basename(run_dir)})'
    )
    
    report_path = os.path.join(reports_test_dir, 'test_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved unbiased test evaluation files to: {run_dir}")

def run_challenge_test(
    run_dir: str, 
    data_path: str = 'data/paf-prediction-challenge-database/'
):
    """Runs challenge predictions using the model ensemble from a run directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, window_seconds, use_hrv, hrv_mean, hrv_std, k_fold = load_model_from_run(run_dir, device)
    
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
    probabilities = {}

    with torch.no_grad(): 
        for name in test_records:
            try:
                input_tensor, hrv_tensor = load_test_record(
                    data_path, name, window_seconds=window_seconds,
                    hrv_mean=hrv_mean, hrv_std=hrv_std
                )
                input_tensor = input_tensor.to(device)
                hrv_tensor = hrv_tensor.to(device)
                
                probs_list = []
                for model in models:
                    if use_hrv:
                        output = model(input_tensor, hrv_tensor)
                    else:
                        output = model(input_tensor)
                    prob = torch.softmax(output, dim=1) # shape (1, 2)
                    probs_list.append(prob.cpu().numpy())
                
                avg_prob = np.mean(probs_list, axis=0) # shape (1, 2)
                pred_label = np.argmax(avg_prob[0])
                confidence = avg_prob[0][pred_label]

                results[name] = int(pred_label)
                probabilities[name] = float(avg_prob[0][1])
                print(f"Record {name}: {'PRE-PAF' if pred_label == 1 else 'NORMAL'} (Conf: {confidence:.2f})")
            
            except Exception as e:
                print(f"Error processing challenge record {name}: {e}")

    print("\n--- Challenge Evaluation Summary (Subject Pairs) ---")
    for i in range(1, len(test_records), 2):
        if i < len(test_records):
            r1, r2 = test_records[i-1], test_records[i]
            print(f"Pair {r1}/{r2} Predictions: {results.get(r1, 'N/A')} / {results.get(r2, 'N/A')}")
            
    # Save predictions & probabilities in the structured reports/test/ folder
    reports_test_dir = os.path.join(run_dir, "reports", "test")
    os.makedirs(reports_test_dir, exist_ok=True)
    
    predictions_path = os.path.join(reports_test_dir, "challenge_predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(results, f, indent=4)
        
    probs_path = os.path.join(reports_test_dir, "challenge_probabilities.json")
    with open(probs_path, "w") as f:
        json.dump(probabilities, f, indent=4)
        
    print(f"Saved challenge predictions to: {predictions_path}")
    print(f"Saved challenge probabilities to: {probs_path}")

def score_and_rename_run_dir(run_dir: str, answers_path: str = 'event-2-answers') -> str:
    """
    Computes the Event 2 challenge score from the saved probabilities and the answers file.
    Then, renames the run folder to append '_scoreXX' where XX is the adjusted score percentage.
    Returns the new path of the run folder.
    """
    import re
    
    reports_test_dir = os.path.join(run_dir, "reports", "test")
    probs_path = os.path.join(reports_test_dir, "challenge_probabilities.json")
    
    if not os.path.exists(probs_path):
        print(f"No challenge probabilities file found at {probs_path}. Cannot score Event 2.")
        return run_dir
        
    if not os.path.exists(answers_path):
        print(f"No answers key found at {answers_path}. Skipping Event 2 scoring.")
        return run_dir
        
    # 1. Parse answers
    answers = {}
    with open(answers_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                rec, lbl = parts
                answers[rec] = lbl
                
    # 2. Load probabilities
    with open(probs_path, 'r') as f:
        probabilities = json.load(f)
        
    # 3. Evaluate pairs
    raw_score = 0
    group_a_correct = 0
    group_a_total = 0
    
    sorted_records = sorted(list(probabilities.keys()))
    if len(sorted_records) < 100:
        print(f"Warning: Only {len(sorted_records)} challenge records found in probabilities. Event 2 requires all 100 records.")
        return run_dir
        
    for i in range(1, 101, 2):
        r1 = f"t{i:02d}"
        r2 = f"t{i+1:02d}"
        
        lbl1 = answers.get(r1, 'N')
        lbl2 = answers.get(r2, 'N')
        
        p1 = probabilities.get(r1, 0.5)
        p2 = probabilities.get(r2, 0.5)
        
        is_group_a = (lbl1 == 'A' or lbl2 == 'A')
        
        if not is_group_a:
            raw_score += 1
        else:
            group_a_total += 1
            if p1 > p2:
                pred1, pred2 = 'A', 'N'
            else:
                pred1, pred2 = 'N', 'A'
                
            if pred1 == lbl1 and pred2 == lbl2:
                group_a_correct += 1
                raw_score += 1
                
    adjusted_score = raw_score - 22
    pct_score = int(round(adjusted_score / 28 * 100))
    
    print("\n" + "="*50)
    print("EVENT 2 OFFICIAL CHALLENGE EVALUATION")
    print("="*50)
    print(f"Group A correct pairs: {group_a_correct} / {group_a_total} ({group_a_correct/group_a_total*100:.2f}%)")
    print(f"Raw Event 2 score: {raw_score} / 50")
    print(f"Adjusted Event 2 score: {adjusted_score} / 28 ({pct_score}%)")
    print(f"Top challenge leaderboard: 1st: 79% (22/28) | 2nd: 71% (20/28)")
    print("="*50 + "\n")
    
    # 4. Rename folder to append _scoreXX
    base_dir = run_dir.rstrip('/\\')
    base_dir_clean = re.sub(r'_score\d+$', '', base_dir)
    new_run_dir = f"{base_dir_clean}_score{pct_score}"
    
    if run_dir != new_run_dir:
        # Check if destination already exists, rename it (e.g. if we are overwriting/re-running)
        if os.path.exists(new_run_dir):
            import shutil
            try:
                shutil.rmtree(new_run_dir)
            except Exception:
                pass
        try:
            os.rename(run_dir, new_run_dir)
            print(f"Renamed run directory: {run_dir} -> {new_run_dir}")
            return new_run_dir
        except Exception as e:
            print(f"Warning: Failed to rename directory to {new_run_dir}: {e}")
            return run_dir
    else:
        return run_dir
