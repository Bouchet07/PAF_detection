from data_manager import get_loaders
from model import PAFClassifier
import torch
import torch.nn as nn
import torch.optim as optim

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = get_loaders()
    model = PAFClassifier().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    
    model.train() # Set model to training mode
    print(f"Training starting on {device}...")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            # Move data to GPU/CPU
            signals, labels = signals.to(device), labels.to(device)
            
            # --- The Core Training Steps ---
            optimizer.zero_grad()               # Clear previous gradients
            outputs = model(signals)            # Forward pass
            loss = criterion(outputs, labels)   # Calculate loss
            loss.backward()                     # Backward pass (compute gradients)
            optimizer.step()                    # Update weights
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

        print(f"Epoch {epoch+1} complete. Average Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "paf_resnet.pth")
    print("Model saved!")

# IMPORTANT: Windows still needs this guard here!
if __name__ == "__main__":
    train()