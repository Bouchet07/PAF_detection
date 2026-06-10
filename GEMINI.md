# PAF Detection Project

This project aims to predict Paroxysmal Atrial Fibrillation (PAF) from ECG signals using deep learning. Specifically, it focuses on predicting PAF episodes within a **5-minute window before they occur**.

## Project Overview

- **Objective:** Predict PAF episodes 5 minutes before onset from ECG recordings.
- **Main Technologies:**
    - **Language:** Python 3.13+
    - **Deep Learning:** PyTorch (1D ResNet).
    - **Signal Processing:** `wfdb` for reading PhysioNet signals.
    - **Data Handling:** NumPy, Scikit-learn.
    - **Dependency Management:** `uv`.

## Datasets

### 1. PAF Prediction Challenge Database (afpdb)
- **Source:** [PhysioNet - PAF Prediction Challenge](https://physionet.org/content/afpdb/1.0.0/)
- **Records:**
    - `n`: 30-minute recordings from patients without PAF (Normal/Distant).
    - `p`: 30-minute recordings immediately preceding a PAF episode (Pre-PAF).
    - `t`: Test records for challenge evaluation.
- **Task:** Classify if a recording is from a patient at risk of PAF and whether PAF is imminent.

### 2. CPSC 2021 (China Physiological Signal Challenge)
- **Source:** [PhysioNet - CPSC 2021](https://physionet.org/content/cpsc2021/1.0.0/)
- **Description:** Focuses on detecting PAF events (onset and end) in dynamic ECG recordings.
- **Labels:** 
    - Non-AF rhythm (N).
    - Persistent AF rhythm (AFf).
    - Paroxysmal AF rhythm (AFp).

## Architecture

The project has been refactored into a clean, modular structure:

- `src/data/`:
    - `preprocess.py`: Cleans and preprocesses all four databases (`afpdb`, `cpsc2021`, `ltafdb`, and `shdb-af`). Automatically standardizes channels (2), sampling rate (128Hz), and filters out invalid overlapping AFIB pre-onset segments to prevent data leakage.
    - `dataset.py`: Defines the `PAFDataset` PyTorch class with channel-wise Z-score normalization and dynamic windowing.
    - `dataloader.py`: Handles train/val/test subject-grouped splitting and creates PyTorch DataLoaders.
- `src/models/`:
    - `resnet1d.py`: Defines the baseline `PAFClassifier` (1D ResNet).
    - `transformer1d.py`: Defines the `CNNTransformerPAFClassifier` (hybrid CNN-Transformer for temporal correlations).
    - `senet1d.py`: Defines the `SEResNetPAFClassifier` (1D ResNet with Squeeze-and-Excitation channel attention blocks).
- `src/utils/`:
    - `metrics.py`: Contains utility functions for plotting training loss/F1 curves and confusion matrices.
    - `leakage.py`: Checks for subject or record leakage between splits (run with `python -m src.utils.leakage`).
    - `stats.py`: Prints statistics across all database sources (run with `python -m src.utils.stats`).
    - `download.py`: CLI download manager for database retrieval (run with `python -m src.utils.download`).

Root Runner Scripts:
- `train.py`: CLI runner to start model training.
- `test.py`: CLI runner to evaluate the model on the validation set and run challenge predictions.

## Development Strategy: 5-Minute Pre-PAF Prediction

The primary goal is to adapt the data loading and model to focus on the 5-minute window prior to PAF onset.
- **Windowing:** Instead of random 30s slices from the entire 30m record, the focus will be on the segments closest to the onset (for `p` records).
- **Targeting:** Clarify if "5-minute window" refers to the input duration or the lead time before onset.

## Building and Running

### Setup
```bash
uv sync
uv sync --extra gpu  # For CUDA support
```

### Data Acquisition & Preprocessing
```bash
uv run python -m src.utils.download
uv run python -m src.data.preprocess
```

### Training & Evaluation
```bash
uv run python train.py
uv run python test.py
```

## Conventions

- **Subject Grouping:** Always group records by subject (e.g., `p15` and `p15c` belong together) to prevent data leakage during train/val split.
- **Input Shape:** `(Batch, 2, Samples)`.
- **Normalization:** Z-score normalization per channel.
