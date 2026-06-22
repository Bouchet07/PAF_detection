import subprocess
import os

# Define the Transformer models to train
model_types = [
    "transformer",          # Baseline CNN-Transformer
    "transformer_deep",     # Deeper Transformer (6 encoder layers)
    "transformer_wide",     # Wider Transformer (d_model=256, nhead=8, ff=512)
    "transformer_dropout",  # Higher dropout regularisation (dropout=0.4)
    "transformer_narrow"    # Narrower Transformer (d_model=64, nhead=2, ff=128)
]

results_subfolder = "transformer_variations"
learning_rate = "0.0003"  # Stable learning rate for Attention blocks

print("="*80)
print("STARTING TRANSFORMER 1D ARCHITECTURE STUDY")
print("="*80)

for model_type in model_types:
    for augment_flag in [False, True]:
        aug_suffix = "aug" if augment_flag else "no_aug"
        print("\n" + "="*80)
        print(f"STARTING EXPERIMENT: {model_type} ({aug_suffix})")
        print("="*80 + "\n")
        
        # Save folder name: e.g., results/transformer_variations/transformer_no_aug
        run_name = f"{results_subfolder}/{model_type}_{aug_suffix}"
        
        # 1. Train the model using 5-fold Group CV with HRV
        train_cmd = [
            "uv", "run", "python", "train.py",
            "--model_type", model_type,
            "--window_seconds", "10",
            "--num_epochs", "30",
            "--lr", learning_rate,
            "--use_hrv",
            "--k_fold", "5",
            "--run_name", run_name
        ]
        
        if augment_flag:
            train_cmd.append("--augment")
            
        print(f"Running train command: {' '.join(train_cmd)}")
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {model_type}_{aug_suffix}: {e}")
            continue
            
        # 2. Evaluate on test set, perform challenge predictions, and rename directory with score
        run_dir = os.path.join("results", results_subfolder, f"{model_type}_{aug_suffix}")
        test_cmd = [
            "uv", "run", "python", "test.py",
            "--run_dir", run_dir
        ]
        
        print(f"Running evaluation command: {' '.join(test_cmd)}")
        try:
            subprocess.run(test_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model_type}_{aug_suffix}: {e}")
            continue
            
        print(f"Successfully finished experiment for: {model_type}_{aug_suffix}\n")

print("="*80)
print("ALL TRANSFORMER VARIATIONS TRAINED AND EVALUATED!")
print("="*80)
