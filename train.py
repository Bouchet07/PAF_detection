from data_manager import get_loaders
from model import PAFClassifier
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = get_loaders()
    model = PAFClassifier().to(device)
    
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