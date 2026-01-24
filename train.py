from data_manager import get_loaders
from model import ECGClassifier
import torch

def train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the loader via our factory function
    train_loader = get_loaders()
    model = ECGClassifier().to(device)
    
    # Dummy Training Loop
    print(f"Training starting on {device}...")
    for signals, labels in train_loader:
        signals, labels = signals.to(device), labels.to(device)
        
        # Zero gradients, forward pass, backward pass, etc...
        print(f"Processing batch of {signals.shape[0]} signals")
        break 

# IMPORTANT: Windows still needs this guard here!
if __name__ == "__main__":
    train()