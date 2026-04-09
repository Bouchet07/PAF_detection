import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_manager import get_loaders
from model import PAFClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_loaders()
    model = PAFClassifier().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    num_epochs = 50
    best_val_acc = 0.0
    
    # Initialize a dictionary to store all our metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"Training starting on {device}...")
    print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Train Acc':<12} | {'Val Acc':<12}")
    print("-" * 65)
    
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train() 
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()               
            outputs = model(signals)            
            loss = criterion(outputs, labels)   
            loss.backward()                     
            optimizer.step()                    
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                
                # Calculate validation loss
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        scheduler.step(epoch_val_loss)
        
        # --- RECORD METRICS ---
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print clearly formatted row
        print(f"[{epoch+1:02d}/{num_epochs:02d}]    | "
              f"{epoch_train_loss:.4f}       | "
              f"{epoch_val_loss:.4f}     | "
              f"{epoch_train_acc:05.2f}%      | "
              f"{epoch_val_acc:05.2f}%")

        # Save the model ONLY if it's the best one we've seen so far
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "paf_resnet_best.pth")

    print("-" * 65)
    print(f"Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Best model saved as 'paf_resnet_best.pth'")
    
    # --- PLOTTING ---
    plot_training_curves(history)

def plot_training_curves(history):
    """Generates and saves a plot of the training history."""
    print("Generating training curves plot...")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o', markersize=4)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy', marker='o', markersize=4)
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o', markersize=4)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot to a file so you can view it after the run
    plt.savefig('training_history.png', dpi=300)
    print("Plot saved as 'training_history.png'")
    
    # If running in an environment that supports it (like Jupyter), this displays the plot
    plt.show()

if __name__ == "__main__":
    train()