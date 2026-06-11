import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(history: dict, save_path: str = 'training_history.png'):
    """
    Plots training and validation loss, macro F1-score & accuracy curves.
    Supports K-Fold curves (list of lists) by plotting the mean and standard deviation shading.
    """
    # Detect if history has list of lists
    is_kfold = len(history['train_loss']) > 0 and isinstance(history['train_loss'][0], (list, np.ndarray))
    
    if is_kfold:
        # Pad histories to the maximum epoch length using final value carryforward
        max_epochs = max(len(h) for h in history['train_loss'])
        epochs = range(1, max_epochs + 1)
        
        def pad_history(metric_list):
            padded = []
            for h in metric_list:
                if len(h) < max_epochs:
                    last_val = h[-1]
                    h = list(h) + [last_val] * (max_epochs - len(h))
                padded.append(h)
            return np.array(padded)
            
        train_loss = pad_history(history['train_loss'])
        val_loss = pad_history(history['val_loss'])
        train_f1 = pad_history(history['train_f1'])
        val_f1 = pad_history(history['val_f1'])
        
        has_accuracy = 'train_acc' in history and 'val_acc' in history
        if has_accuracy:
            train_acc = pad_history(history['train_acc'])
            val_acc = pad_history(history['val_acc'])
    else:
        epochs = range(1, len(history['train_loss']) + 1)
        has_accuracy = 'train_acc' in history and 'val_acc' in history
        
    if has_accuracy:
        plt.figure(figsize=(20, 5))
        num_subplots = 3
    else:
        plt.figure(figsize=(14, 5))
        num_subplots = 2
        
    def plot_metric(subplot_idx, train_vals, val_vals, title, ylabel, train_color, val_color):
        plt.subplot(1, num_subplots, subplot_idx)
        if is_kfold:
            # Train Stats
            train_mean = np.mean(train_vals, axis=0)
            train_std = np.std(train_vals, axis=0)
            plt.plot(epochs, train_mean, label='Train Mean', color=train_color, linewidth=2)
            plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15, color=train_color)
            
            # Val Stats
            val_mean = np.mean(val_vals, axis=0)
            val_std = np.std(val_vals, axis=0)
            plt.plot(epochs, val_mean, label='Val Mean', color=val_color, linewidth=2)
            plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15, color=val_color)
        else:
            plt.plot(epochs, train_vals, label='Train', color=train_color, linewidth=2)
            plt.plot(epochs, val_vals, label='Val', color=val_color, linewidth=2)
            
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
    # Plot subplots
    if is_kfold:
        plot_metric(1, train_loss, val_loss, 'Training & Validation Loss (K-Fold)', 'Loss', '#1f77b4', '#ff7f0e')
        plot_metric(2, train_f1, val_f1, 'Training & Validation Macro F1 (K-Fold)', 'Macro F1', '#2ca02c', '#d62728')
        if has_accuracy:
            plot_metric(3, train_acc, val_acc, 'Training & Validation Accuracy (K-Fold)', 'Accuracy', '#9467bd', '#8c564b')
    else:
        plot_metric(1, history['train_loss'], history['val_loss'], 'Training & Validation Loss', 'Loss', '#1f77b4', '#ff7f0e')
        plot_metric(2, history['train_f1'], history['val_f1'], 'Training & Validation Macro F1 Score', 'Macro F1', '#2ca02c', '#d62728')
        if has_accuracy:
            plot_metric(3, history['train_acc'], history['val_acc'], 'Training & Validation Accuracy', 'Accuracy', '#9467bd', '#8c564b')
            
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


def plot_eval_curves(y_true: np.ndarray, y_probs: np.ndarray, save_path: str = 'evaluation_curves.png', title_prefix: str = ''):
    """
    Plots ROC and Precision-Recall (PR) curves side-by-side.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(16, 6))
    
    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title(f'{title_prefix} ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='#2ca02c', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})\nAP = {ap:.4f}')
    # No-skill baseline is the ratio of positive class samples
    baseline = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
    plt.plot([0, 1], [baseline, baseline], color='grey', lw=1, linestyle='--', label=f'Baseline ({baseline:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title(f'{title_prefix} Precision-Recall Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure save directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Evaluation curves saved to {save_path} (ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f})")
    return roc_auc, pr_auc

