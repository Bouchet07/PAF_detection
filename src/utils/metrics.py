import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(history: dict, save_path: str = 'training_history.png'):
    """
    Plots training and validation loss, macro F1-score & accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    has_accuracy = 'train_acc' in history and 'val_acc' in history
    
    if has_accuracy:
        plt.figure(figsize=(20, 5))
        num_subplots = 3
    else:
        plt.figure(figsize=(14, 5))
        num_subplots = 2
        
    # Loss plot
    plt.subplot(1, num_subplots, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # F1 score plot
    plt.subplot(1, num_subplots, 2)
    plt.plot(epochs, history['train_f1'], label='Train F1', color='#2ca02c', linewidth=2)
    plt.plot(epochs, history['val_f1'], label='Val F1', color='#d62728', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title('Training & Validation Macro F1 Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Accuracy plot if available
    if has_accuracy:
        plt.subplot(1, num_subplots, 3)
        plt.plot(epochs, history['train_acc'], label='Train Acc', color='#9467bd', linewidth=2)
        plt.plot(epochs, history['val_acc'], label='Val Acc', color='#8c564b', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = 'confusion_matrix.png', title: str = 'Confusion Matrix'):
    """
    Plots a heatmapped confusion matrix with Seaborn.
    """
    plt.figure(figsize=(8, 6))
    target_names = ['Normal/Distant', 'Pre-PAF']
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=target_names, 
        yticklabels=target_names,
        annot_kws={"size": 12}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Ensure save directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
