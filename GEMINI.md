# PAF Detection Project

This project aims to predict Paroxysmal Atrial Fibrillation (PAF) from ECG signals using deep learning. It implements a ResNet-like architecture for 1D signal classification.

## Project Overview

- **Objective:** Detect/predict PAF episodes from ECG recordings.
- **Main Technologies:**
    - **Language:** Python 3.13+
    - **Deep Learning:** PyTorch (1D Convolutional Neural Networks with Residual Blocks).
    - **Signal Processing:** `wfdb` (Waveform Database) for reading physiological signals from PhysioNet.
    - **Data Handling:** NumPy, Scikit-learn.
    - **Visualization:** Matplotlib, Seaborn.
    - **Dependency Management:** `uv`.

## Architecture

- `data_manager.py`: Handles loading WFDB records into memory and provides a PyTorch `Dataset` (`PAFDataset`) and `DataLoader`. It performs random slicing of signals into windows (default 30s at 128Hz).
- `model.py`: Defines the `PAFClassifier`, a 1D ResNet architecture consisting of `ResidualBlock` modules.
- `train.py`: The training script. It uses `CrossEntropyLoss` and the `Adam` optimizer. Saves the trained model to `paf_resnet.pth`.
- `test.py`: Evaluation script. Includes functions for inference on challenge test records and generating classification reports/confusion matrices.
- `download-manager.py`: An interactive CLI tool to download various ECG datasets from PhysioNet (e.g., `afpdb`, `mitdb`, `cpsc2021`).
- `explore.py`: Utility for data exploration (implied by filename).
- `2d.py`: Likely an experimental script for 2D representation of ECG data.

## Building and Running

### Prerequisites
Ensure you have `uv` installed.

### Setup
Install dependencies:
```bash
uv sync
```
To include GPU support (CUDA 13.0):
```bash
uv sync --extra gpu
```

### Data Acquisition
Download the necessary datasets:
```bash
uv run download-manager.py
```
Datasets are stored in the `data/` directory.

### Training
Start the training process:
```bash
uv run train.py
```
The model weights will be saved as `paf_resnet.pth`.

### Evaluation
Run evaluation and generate metrics:
```bash
uv run test.py
```

## Development Conventions

- **Data Loading:** The `PAFDataset` loads the entire dataset into RAM. Ensure your system has sufficient memory or modify `data_manager.py` for large datasets.
- **Model Inputs:** The model expects input shapes of `(Batch, Channels, Samples)`. For the default configuration, this is `(Batch, 2, 3840)` (30 seconds at 128Hz).
- **Labeling:**
    - `0`: Normal/Distant (recordings starting with 'n').
    - `1`: Pre-PAF (recordings starting with 'p').
- **Testing:** The `test.py` script specifically handles the "t" records from the PAF Prediction Challenge, pairing them (e.g., t01/t02) as per challenge rules.
