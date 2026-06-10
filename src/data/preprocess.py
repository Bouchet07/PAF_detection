import os
import re
import numpy as np
import pandas as pd
import wfdb
import scipy.signal
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.interpolate import interp1d

TARGET_FS = 128
WINDOW_PRE_PAF_SEC = 300  # 5-minute pre-onset window
CHANNELS = 2

@dataclass
class WindowMetadata:
    filename: str
    subject: str
    label: int
    source: str
    mean_rr: float = 0.8
    std_rr: float = 0.05
    rmssd: float = 0.03
    pnn50: float = 5.0
    mean_hr: float = 75.0
    std_hr: float = 5.0
    lf: float = 0.0
    hf: float = 0.0
    lf_hf_ratio: float = 1.0

def compute_hrv_features(r_peaks: List[int], fs: float) -> dict:
    if len(r_peaks) < 10:
        return {
            'mean_rr': 0.8, 'std_rr': 0.05, 'rmssd': 0.03, 'pnn50': 5.0, 
            'mean_hr': 75.0, 'std_hr': 5.0, 'lf': 0.0, 'hf': 0.0, 'lf_hf_ratio': 1.0
        }
        
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    pnn50 = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr) * 100.0
    hr = 60.0 / rr_intervals
    mean_hr = np.mean(hr)
    std_hr = np.std(hr)
    
    try:
        t = np.cumsum(rr_intervals)
        t = t - t[0]
        fs_interp = 4.0
        t_interp = np.arange(0, t[-1], 1.0 / fs_interp)
        
        if len(t_interp) > 32:
            f_interp = interp1d(t, rr_intervals, kind='linear', fill_value='extrapolate')
            rr_interp = f_interp(t_interp)
            rr_interp = rr_interp - np.mean(rr_interp)
            
            f, psd = scipy.signal.welch(rr_interp, fs=fs_interp, nperseg=min(256, len(rr_interp)))
            lf_band = (f >= 0.04) & (f < 0.15)
            hf_band = (f >= 0.15) & (f <= 0.4)
            
            def trapz(y, x):
                if len(y) < 2: return 0.0
                return np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0)
                
            lf = trapz(psd[lf_band], f[lf_band]) if np.any(lf_band) else 0.0
            hf = trapz(psd[hf_band], f[hf_band]) if np.any(hf_band) else 0.0
            lf_hf_ratio = lf / (hf + 1e-8) if hf > 0 else 1.0
        else:
            lf, hf, lf_hf_ratio = 0.0, 0.0, 1.0
    except Exception:
        lf, hf, lf_hf_ratio = 0.0, 0.0, 1.0
        
    return {
        'mean_rr': float(mean_rr),
        'std_rr': float(std_rr),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50),
        'mean_hr': float(mean_hr),
        'std_hr': float(std_hr),
        'lf': float(lf),
        'hf': float(hf),
        'lf_hf_ratio': float(lf_hf_ratio)
    }

def find_records(directory: str) -> List[str]:
    """Recursively finds all WFDB record paths (without extensions) in directory."""
    records = []
    if not os.path.exists(directory):
        return []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.hea'):
                record_path = os.path.join(root, f[:-4])
                # Ensure the corresponding .dat file exists
                if os.path.exists(record_path + '.dat'):
                    records.append(record_path)
    return sorted(records)

def get_shdb_subject_mapping(shdb_dir: str) -> Dict[str, str]:
    """Parses AdditionalData.csv in SHDB-AF to map Data_ID (record number) to Subject_ID."""
    csv_path = os.path.join(shdb_dir, "AdditionalData.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(shdb_dir, "shdb-af", "1.0.1", "AdditionalData.csv") # Alternate nested path check
        if not os.path.exists(csv_path):
            return {}
            
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        mapping = {}
        for _, row in df.iterrows():
            if 'Data_ID' in row and 'Subject_ID' in row:
                data_id = str(row['Data_ID']).strip().zfill(3)
                subj_id = str(row['Subject_ID']).strip()
                mapping[data_id] = f"shdb_{subj_id}"
        return mapping
    except Exception as e:
        print(f"Warning: Failed to load SHDB subject mapping: {e}")
        return {}

def preprocess_afpdb(data_path: str, output_dir: str) -> List[WindowMetadata]:
    """
    Preprocesses AFPDB (PAF Prediction Challenge Database).
    - p records: Last 5 minutes represents Pre-PAF (Label 1).
    - n records: Take 3 non-overlapping 5-minute segments (Label 0).
    - Group by subject pair to prevent leakage.
    """
    records = find_records(data_path)
    metadata = []
    samples_needed = WINDOW_PRE_PAF_SEC * TARGET_FS

    for r_path in tqdm(records, desc="Preprocessing AFPDB"):
        r_name = os.path.basename(r_path)
        try:
            if r_name.startswith('n'):
                label = 0
                subject_prefix = 'n'
            elif r_name.startswith('p') and not r_name.endswith('c'):
                label = 1
                subject_prefix = 'p'
            else:
                # Skip continuation ('c') and test ('t') records
                continue

            record = wfdb.rdrecord(r_path)
            signal = record.p_signal.astype(np.float32).T

            # Force 2 channels
            if signal.shape[0] < CHANNELS:
                padding = np.zeros((CHANNELS - signal.shape[0], signal.shape[1]), dtype=np.float32)
                signal = np.vstack([signal, padding])
            else:
                signal = signal[:CHANNELS, :]

            # Resample to target fs if needed
            if record.fs != TARGET_FS:
                num_samples = int(signal.shape[1] * TARGET_FS / record.fs)
                signal = scipy.signal.resample(signal, num_samples, axis=1).astype(np.float32)

            # Extract subject pair to group records
            num = int(re.search(r'\d+', r_name).group())
            pair_id = (num - 1) // 2
            subject_id = f"afpdb_{subject_prefix}_{pair_id}"

            # Load R-peaks from QRS annotation
            try:
                ann = wfdb.rdann(r_path, 'qrs')
                r_peaks = ann.sample
            except Exception:
                r_peaks = []
                
            scale_factor = TARGET_FS / record.fs

            if label == 1:
                # Pre-PAF: Grab the last 5 minutes of the record
                start = max(0, signal.shape[1] - samples_needed)
                segment = signal[:, start:]
                if segment.shape[1] < samples_needed:
                    segment = np.pad(segment, ((0,0), (0, samples_needed - segment.shape[1])), 'constant')
                
                # Convert segment bounds to original sampling rate space
                start_orig = int(start / scale_factor)
                end_orig = int(record.sig_len)
                
                peaks_in_seg = [p - start_orig for p in r_peaks if start_orig <= p < end_orig]
                rescaled_peaks = [int(p * scale_factor) for p in peaks_in_seg]
                hrv = compute_hrv_features(rescaled_peaks, TARGET_FS)
                
                out_name = f"afpdb_{r_name}_pre.npy"
                np.save(os.path.join(output_dir, out_name), segment)
                metadata.append(WindowMetadata(out_name, subject_id, label, "afpdb", **hrv))
            else:
                # Normal: Extract up to 3 non-overlapping 5-min segments
                for i in range(3):
                    start = i * samples_needed
                    if start + samples_needed > signal.shape[1]: 
                        break
                    segment = signal[:, start:start+samples_needed]
                    
                    # Convert segment bounds to original sampling rate space
                    start_orig = int(start / scale_factor)
                    end_orig = int((start + samples_needed) / scale_factor)
                    
                    peaks_in_seg = [p - start_orig for p in r_peaks if start_orig <= p < end_orig]
                    rescaled_peaks = [int(p * scale_factor) for p in peaks_in_seg]
                    hrv = compute_hrv_features(rescaled_peaks, TARGET_FS)
                    
                    out_name = f"afpdb_{r_name}_norm_{i}.npy"
                    np.save(os.path.join(output_dir, out_name), segment)
                    metadata.append(WindowMetadata(out_name, subject_id, label, "afpdb", **hrv))

        except Exception as e:
            print(f"Error processing AFPDB record {r_name}: {e}")

    return metadata

def preprocess_wfdb_dataset(
    data_path: str, 
    dataset_name: str, 
    output_dir: str, 
    subject_mapping: Optional[Dict[str, str]] = None,
    max_control_segments: int = 3,
    label_0_gap_sec: int = 1800
) -> List[WindowMetadata]:
    """
    Preprocesses dynamic ECG datasets with annotations (CPSC2021, LTAFDB, SHDB-AF).
    Extracts:
    - Label 1: 5-minute pre-PAF window immediately before AFIB onsets (ensures no AFIB inside the window).
    - Label 0: 5-minute normal segments that are clean and at least `label_0_gap_sec` away from any AFIB episode.
    """
    records = find_records(data_path)
    metadata = []
    pre_paf_samples = WINDOW_PRE_PAF_SEC * TARGET_FS
    gap_samples = label_0_gap_sec * TARGET_FS

    for r_path in tqdm(records, desc=f"Preprocessing {dataset_name.upper()}"):
        r_name = os.path.basename(r_path)
        atr_path = r_path + '.atr'
        
        # We only process annotated records (must have .atr file)
        if not os.path.exists(atr_path):
            continue

        try:
            # Parse Subject ID
            if dataset_name == 'cpsc':
                # record format: record_01_1
                subject_id = f"cpsc_{r_name.split('_')[1]}"
            elif dataset_name == 'shdb-af' and subject_mapping:
                # mapping key is e.g. '001'
                subject_id = subject_mapping.get(r_name, f"shdb_{r_name}")
            else:
                subject_id = f"{dataset_name}_{r_name}"

            record = wfdb.rdrecord(r_path)
            ann = wfdb.rdann(r_path, 'atr')
            fs = record.fs
            
            signal = record.p_signal.astype(np.float32).T

            # Force 2 channels
            if signal.shape[0] < CHANNELS:
                padding = np.zeros((CHANNELS - signal.shape[0], signal.shape[1]), dtype=np.float32)
                signal = np.vstack([signal, padding])
            else:
                signal = signal[:CHANNELS, :]

            # Resample signal to target frequency
            if fs != TARGET_FS:
                num_samples = int(signal.shape[1] * TARGET_FS / fs)
                signal = scipy.signal.resample(signal, num_samples, axis=1).astype(np.float32)
                scale_factor = TARGET_FS / fs
            else:
                scale_factor = 1.0

            total_samples = signal.shape[1]

            # Parse annotation sample indices adjusted for resampling
            r_samples = [int(s * scale_factor) for s in ann.sample]
            r_notes = ann.aux_note

            # Extract rhythm intervals
            intervals = []
            current_rhythm = 'N'
            current_start = 0

            for sample, note in zip(r_samples, r_notes):
                if note.startswith('('):
                    rhythm = note[1:]
                    if rhythm != current_rhythm:
                        if sample > current_start:
                            intervals.append((current_start, min(sample, total_samples), current_rhythm))
                        current_rhythm = rhythm
                        current_start = sample
            intervals.append((current_start, total_samples, current_rhythm))

            # Identify AFIB intervals
            afib_intervals = [(start, end) for start, end, rhythm in intervals if rhythm == 'AFIB']

            # --- 1. Label 1: Pre-PAF Window Extraction ---
            onset_idx = 0
            for start, end in afib_intervals:
                if start < pre_paf_samples:
                    continue  # Not enough data before onset

                # Check if there is any AFIB in the pre-onset window [start - pre_paf_samples, start)
                window_start = start - pre_paf_samples
                window_end = start

                has_afib_in_window = False
                for a_start, a_end in afib_intervals:
                    # Check overlap of AFIB with our 5-minute pre-onset window
                    if max(a_start, window_start) < min(a_end, window_end):
                        has_afib_in_window = True
                        break

                if has_afib_in_window:
                    continue  # Discard if AFIB is present in the pre-onset window

                # Extract and pad if needed
                segment = signal[:, window_start:window_end]
                if segment.shape[1] < pre_paf_samples:
                    segment = np.pad(segment, ((0,0), (0, pre_paf_samples - segment.shape[1])), 'constant')
                elif segment.shape[1] > pre_paf_samples:
                    segment = segment[:, :pre_paf_samples]

                # Convert segment bounds to original sampling rate space
                start_orig = int(window_start / scale_factor)
                end_orig = int(window_end / scale_factor)
                peaks_in_seg = [p - start_orig for p in ann.sample if start_orig <= p < end_orig]
                rescaled_peaks = [int(p * scale_factor) for p in peaks_in_seg]
                hrv = compute_hrv_features(rescaled_peaks, TARGET_FS)

                out_name = f"{dataset_name}_{r_name}_onset_{onset_idx}.npy"
                np.save(os.path.join(output_dir, out_name), segment)
                metadata.append(WindowMetadata(out_name, subject_id, 1, dataset_name, **hrv))
                onset_idx += 1

            # --- 2. Label 0: Normal sinus rhythm window extraction ---
            # Unsafe regions are AFIB episodes +/- gap duration (e.g. 30 minutes)
            unsafe_regions = []
            for a_start, a_end in afib_intervals:
                unsafe_regions.append((max(0, a_start - gap_samples), min(total_samples, a_end + gap_samples)))

            # Merge overlapping unsafe regions
            unsafe_regions = sorted(unsafe_regions, key=lambda x: x[0])
            merged_unsafe = []
            for reg in unsafe_regions:
                if not merged_unsafe:
                    merged_unsafe.append(reg)
                else:
                    prev = merged_unsafe[-1]
                    if reg[0] <= prev[1]:
                        merged_unsafe[-1] = (prev[0], max(prev[1], reg[1]))
                    else:
                        merged_unsafe.append(reg)

            # Safe intervals (complement of unsafe)
            safe_intervals = []
            curr = 0
            for u_start, u_end in merged_unsafe:
                if u_start > curr:
                    safe_intervals.append((curr, u_start))
                curr = max(curr, u_end)
            if curr < total_samples:
                safe_intervals.append((curr, total_samples))

            # Extract control segments from safe intervals
            control_idx = 0
            for s_start, s_end in safe_intervals:
                if control_idx >= max_control_segments:
                    break
                curr_pos = s_start
                while curr_pos + pre_paf_samples <= s_end and control_idx < max_control_segments:
                    segment = signal[:, curr_pos:curr_pos + pre_paf_samples]
                    
                    # Convert segment bounds to original sampling rate space
                    start_orig = int(curr_pos / scale_factor)
                    end_orig = int((curr_pos + pre_paf_samples) / scale_factor)
                    peaks_in_seg = [p - start_orig for p in ann.sample if start_orig <= p < end_orig]
                    rescaled_peaks = [int(p * scale_factor) for p in peaks_in_seg]
                    hrv = compute_hrv_features(rescaled_peaks, TARGET_FS)
                    
                    out_name = f"{dataset_name}_{r_name}_norm_{control_idx}.npy"
                    np.save(os.path.join(output_dir, out_name), segment)
                    metadata.append(WindowMetadata(out_name, subject_id, 0, dataset_name, **hrv))
                    control_idx += 1
                    curr_pos += pre_paf_samples

        except Exception as e:
            print(f"Error processing {dataset_name.upper()} record {r_name}: {e}")

    return metadata

def run_pipeline(data_root="data", output_dir="processed_data", metadata_csv="metadata.csv"):
    """Runs the full preprocessing pipeline for all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_metadata = []

    # 1. AFPDB
    afpdb_path = os.path.join(data_root, "paf-prediction-challenge-database")
    if os.path.exists(afpdb_path):
        meta = preprocess_afpdb(afpdb_path, output_dir)
        all_metadata.extend(meta)
        print(f"Processed AFPDB. Extracted {len(meta)} segments.")
    else:
        print("AFPDB directory not found. Skipping.")

    # 2. CPSC2021
    cpsc_path = os.path.join(data_root, "cpsc2021")
    if os.path.exists(cpsc_path):
        meta = preprocess_wfdb_dataset(cpsc_path, "cpsc", output_dir)
        all_metadata.extend(meta)
        print(f"Processed CPSC2021. Extracted {len(meta)} segments.")
    else:
        print("CPSC2021 directory not found. Skipping.")

    # 3. LTAFDB
    ltafdb_path = os.path.join(data_root, "ltafdb")
    if os.path.exists(ltafdb_path):
        meta = preprocess_wfdb_dataset(ltafdb_path, "ltafdb", output_dir)
        all_metadata.extend(meta)
        print(f"Processed LTAFDB. Extracted {len(meta)} segments.")
    else:
        print("LTAFDB directory not found. Skipping.")

    # 4. SHDB-AF
    shdb_path = os.path.join(data_root, "shdb-af")
    if os.path.exists(shdb_path):
        subject_mapping = get_shdb_subject_mapping(shdb_path)
        meta = preprocess_wfdb_dataset(shdb_path, "shdb-af", output_dir, subject_mapping=subject_mapping)
        all_metadata.extend(meta)
        print(f"Processed SHDB-AF. Extracted {len(meta)} segments.")
    else:
        print("SHDB-AF directory not found. Skipping.")

    # 5. AFDB
    afdb_path = os.path.join(data_root, "afdb")
    if os.path.exists(afdb_path):
        meta = preprocess_wfdb_dataset(afdb_path, "afdb", output_dir)
        all_metadata.extend(meta)
        print(f"Processed AFDB. Extracted {len(meta)} segments.")
    else:
        print("AFDB directory not found. Skipping.")

    # Save metadata
    if all_metadata:
        df = pd.DataFrame([m.__dict__ for m in all_metadata])
        df.to_csv(metadata_csv, index=False)
        print(f"\nPreprocessing finished successfully!")
        print(f"Total segments: {len(df)}")
        print(df['label'].value_counts())
        print(df['source'].value_counts())
    else:
        print("No metadata generated. No data was processed.")

if __name__ == "__main__":
    run_pipeline()
