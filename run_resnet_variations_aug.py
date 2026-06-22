import subprocess
import os

# Define the remaining models to train with augmentation
model_types = [
    "resnet1d",               # Baseline model (layers=[2, 2, 2, 2], channels=[32, 64, 128, 256])
    "resnet1d_deep",          # Deeper ResNet (layers=[3, 3, 3, 3])
    "resnet1d_wide",          # Wider ResNet (channels=[64, 128, 256, 512])
    "resnet1d_large_kernel",  # Larger Conv kernel size (kernel_size=23)
    "resnet1d_small_kernel"   # Smaller Conv kernel size (kernel_size=7)
]

results_subfolder = "resnet_variations"

print("="*80)
print("STARTING AUGMENTED RESNET 1D ARCHITECTURE STUDY")
print("="*80)

for model_type in model_types:
    print("\n" + "="*80)
    print(f"STARTING AUGMENTED EXPERIMENT: {model_type}")
    print("="*80 + "\n")
    
    # Save folder name: e.g., results/resnet_variations/resnet1d_aug
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
print("ALL AUGMENTED VARIATIONS TRAINED AND EVALUATED!")
print("="*80)
