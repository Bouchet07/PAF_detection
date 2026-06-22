import subprocess
import os
import sys

# Define the models we want to evaluate
model_types = [
    "resnet1d_deep",          # Deeper ResNet (layers=[3, 3, 3, 3] instead of [2, 2, 2, 2])
    "resnet1d_wide",          # Wider ResNet (channels=[64, 128, 256, 512] instead of [32, 64, 128, 256])
    "resnet1d_large_kernel",  # Larger Conv kernel size (kernel_size=23 instead of 15)
    "resnet1d_small_kernel",  # Smaller Conv kernel size (kernel_size=7 instead of 15)
    "resnet1d_deep_narrow"    # Deeper but narrower (layers=[3, 4, 6, 3], channels=[24, 48, 96, 192])
]

results_subfolder = "resnet_variations"

print("="*80)
print("STARTING RESNET 1D ARCHITECTURE VARIATIONS STUDY")
print("="*80)

for model_type in model_types:
    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT: {model_type}")
    print("="*80 + "\n")
    
    # Path inside results/resnet_variations/
    run_name = f"{results_subfolder}/{model_type}"
    
    # 1. Train the model using 5-fold Group CV (with HRV)
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
        print(f"Error training {model_type}: {e}")
        continue
        
    # 2. Evaluate on test set, perform challenge predictions, and rename directory with score
    run_dir = os.path.join("results", results_subfolder, model_type)
    test_cmd = [
        "uv", "run", "python", "test.py",
        "--run_dir", run_dir
    ]
    
    print(f"Running evaluation command: {' '.join(test_cmd)}")
    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_type}: {e}")
        continue
        
    print(f"Successfully finished experiment for: {model_type}\n")

print("="*80)
print("ALL ARCHITECTURE VARIATIONS TRAINED AND EVALUATED!")
print("="*80)
