import subprocess
import os

# Define the SEResNet models to train with augmentation
model_types = [
    "senet",               # Baseline SEResNet (layers=[2, 2, 2, 2], channels=[32, 64, 128, 256])
    "senet_deep",          # Deeper SEResNet (layers=[3, 3, 3, 3])
    "senet_wide",          # Wider SEResNet (channels=[64, 128, 256, 512])
    "senet_large_kernel",  # Larger Conv kernel size (kernel_size=23)
    "senet_deep_narrow"    # Deeper but narrower (layers=[3, 4, 6, 3], channels=[24, 48, 96, 192])
]

results_subfolder = "senet_variations"

print("="*80)
print("STARTING AUGMENTED SERESNET 1D ARCHITECTURE STUDY")
print("="*80)

for model_type in model_types:
    print("\n" + "="*80)
    print(f"STARTING AUGMENTED EXPERIMENT: {model_type}")
    print("="*80 + "\n")
    
    # Save folder name: e.g., results/senet_variations/senet_aug
    run_name = f"{results_subfolder}/{model_type}_aug"
    
    # 1. Train the model using 5-fold Group CV with HRV and augmentation
    train_cmd = [
        "uv", "run", "python", "train.py",
        "--model_type", model_type,
        "--window_seconds", "10",
        "--num_epochs", "30",
        "--use_hrv",
        "--k_fold", "5",
        "--augment",
        "--run_name", run_name
    ]
    
    print(f"Running train command: {' '.join(train_cmd)}")
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training {model_type}_aug: {e}")
        continue
        
    # 2. Evaluate on test set, perform challenge predictions, and rename directory with score
    run_dir = os.path.join("results", results_subfolder, f"{model_type}_aug")
    test_cmd = [
        "uv", "run", "python", "test.py",
        "--run_dir", run_dir
    ]
    
    print(f"Running evaluation command: {' '.join(test_cmd)}")
    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_type}_aug: {e}")
        continue
        
    print(f"Successfully finished experiment for: {model_type}_aug\n")

print("="*80)
print("ALL AUGMENTED SERESNET VARIATIONS TRAINED AND EVALUATED!")
print("="*80)
