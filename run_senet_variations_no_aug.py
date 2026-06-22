import subprocess
import os

# Define the SEResNet models to train without augmentation
model_types = [
    "senet",               # Baseline SEResNet
    "senet_deep",          # Deeper SEResNet
    "senet_wide",          # Wider SEResNet
    "senet_large_kernel",  # Larger Conv kernel size (kernel_size=23)
    "senet_small_kernel",  # Smaller Conv kernel size (kernel_size=7)
    "senet_deep_narrow"    # Deeper but narrower
]

results_subfolder = "senet_variations"

print("="*80)
print("STARTING NON-AUGMENTED SERESNET 1D ARCHITECTURE STUDY")
print("="*80)

for model_type in model_types:
    print("\n" + "="*80)
    print(f"STARTING NON-AUGMENTED EXPERIMENT: {model_type}")
    print("="*80 + "\n")
    
    # Save folder name: e.g., results/senet_variations/senet_no_aug
    run_name = f"{results_subfolder}/{model_type}_no_aug"
    
    # 1. Train the model using 5-fold Group CV with HRV and NO augmentation
    train_cmd = [
        "uv", "run", "python", "train.py",
        "--model_type", model_type,
        "--window_seconds", "10",
        "--num_epochs", "30",
        "--use_hrv",
        "--k_fold", "5",
        "--run_name", run_name
    ]
    
    print(f"Running train command: {' '.join(train_cmd)}")
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training {model_type}_no_aug: {e}")
        continue
        
    # 2. Evaluate on test set, perform challenge predictions, and rename directory with score
    run_dir = os.path.join("results", results_subfolder, f"{model_type}_no_aug")
    test_cmd = [
        "uv", "run", "python", "test.py",
        "--run_dir", run_dir
    ]
    
    print(f"Running evaluation command: {' '.join(test_cmd)}")
    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_type}_no_aug: {e}")
        continue
        
    print(f"Successfully finished experiment for: {model_type}_no_aug\n")

print("="*80)
print("ALL NON-AUGMENTED SERESNET VARIATIONS TRAINED AND EVALUATED!")
print("="*80)
