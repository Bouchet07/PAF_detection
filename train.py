import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.dataloader import get_loaders
from src.models import PAFClassifier, CNNTransformerPAFClassifier, SEResNetPAFClassifier
from src.utils.metrics import plot_training_curves, plot_confusion_matrix
from sklearn.metrics import f1_score, classification_report, confusion_matrix

class FocalLoss(nn.Module):
    """
    1D/2D Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(
    model_type: str = 'resnet1d',
    window_seconds: int = 10, 
    batch_size: int = 64, 
    num_epochs: int = 30, 
    lr: float = 0.001,
    metadata_path: str = 'metadata.csv',
    data_dir: str = 'processed_data',
    patience: int = 15,
    run_name: str = None,
    augment: bool = False,
    weight_scheme: str = 'inverse',
    use_sampler: bool = False,
    loss_type: str = 'ce',
    focal_gamma: float = 2.0,
    use_hrv: bool = False,
    resume_from: str = None
):
    """
    Trains the selected 1D ECG model on the preprocessed segments.
    Saves checkpoints, plots, config, and history under results/<run_name>/.
    """
    import json
    import os
    from datetime import datetime
    from sklearn.metrics import accuracy_score
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle resuming from a checkpoint
    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume directory {resume_from} does not exist.")
        config_path = os.path.join(resume_from, "config.json")
        weights_path = os.path.join(resume_from, "best_model.pth")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing config.json or best_model.pth in {resume_from}")
            
        with open(config_path, 'r') as f:
            old_config = json.load(f)
            
        # Override architecture settings to match the loaded weights
        model_type = old_config.get("model_type", model_type)
        use_hrv = old_config.get("use_hrv", use_hrv)
        print(f"Resuming training of {model_type.upper()} from {resume_from} (use_hrv: {use_hrv})")
        
    hrv_dim = 9 if use_hrv else 0
    
    # Validate and instantiate model
    model_type = model_type.lower()
    if model_type == 'resnet1d':
        model = PAFClassifier(in_channels=2, num_classes=2, hrv_dim=hrv_dim).to(device)
    elif model_type == 'transformer':
        model = CNNTransformerPAFClassifier(in_channels=2, num_classes=2, hrv_dim=hrv_dim).to(device)
    elif model_type == 'senet':
        model = SEResNetPAFClassifier(in_channels=2, num_classes=2, hrv_dim=hrv_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Options are 'resnet1d', 'transformer', 'senet'.")
        
    # Load weights if resuming
    if resume_from is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Pretrained weights loaded successfully.")
        
    # 1. Define run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        if resume_from is not None:
            base_run_name = os.path.basename(resume_from.rstrip('/\\'))
            run_name = f"{base_run_name}_continued_{timestamp}"
        else:
            run_name = f"{model_type}_win{window_seconds}_lr{lr}_{timestamp}"
    
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Model: {model_type.upper()}")
    print(f"Saving all outputs to: {run_dir}")
    
    # Get loaders from our modular sub-package (3-way split)
    train_loader, val_loader, test_loader = get_loaders(

        metadata_path=metadata_path,
        data_dir=data_dir,
        batch_size=batch_size, 
        window_seconds=window_seconds,
        augment=augment,
        use_sampler=use_sampler
    )
    
    # Calculate class weights for loss balancing based on scheme
    labels = []
    for _, lbl in train_loader.dataset.metadata.iterrows():
        labels.append(lbl['label'])
    
    class_counts = np.bincount(labels)
    weight_scheme = weight_scheme.lower()
    
    if use_sampler:
        # If using WeightedRandomSampler, the stream of batches is already balanced
        weights = np.ones(len(class_counts), dtype=np.float32)
    elif weight_scheme == 'none':
        weights = np.ones(len(class_counts), dtype=np.float32)
    elif weight_scheme == 'sqrt':
        weights = 1.0 / np.sqrt(class_counts)
        weights = weights / weights.sum() * len(class_counts)
    else:  # 'inverse' or default
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_counts)
        
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Calculated class weights ({weight_scheme}): {weights}")

    # Save initial config with the test set subject list
    test_subjects = test_loader.dataset.metadata['subject'].unique().tolist()
    config = {
        "model_type": model_type,
        "window_seconds": window_seconds,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "patience": patience,
        "device": str(device),
        "class_weights": weights.tolist(),
        "weight_scheme": weight_scheme,
        "augment": augment,
        "use_sampler": use_sampler,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
        "use_hrv": use_hrv,
        "hrv_mean": train_loader.dataset.hrv_mean.tolist() if use_hrv else None,
        "hrv_std": train_loader.dataset.hrv_std.tolist() if use_hrv else None,
        "test_subjects": test_subjects
    }
    
    if loss_type.lower() == 'focal':
        criterion = FocalLoss(alpha=weights_tensor, gamma=focal_gamma)
        print(f"Loss type: Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print("Loss type: Cross Entropy Loss")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print(f"Training starting on {device} (Input window: {window_seconds}s)...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train F1':<9} | {'Val F1':<9} | {'Train Acc':<9} | {'Val Acc':<9}")
    print("-" * 85)
    
    for epoch in range(num_epochs):
        # 1. Training Phase
        model.train() 
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for signals, hrvs, labels in train_loader:
            signals, hrvs, labels = signals.to(device), hrvs.to(device), labels.to(device)
            
            optimizer.zero_grad()               
            if use_hrv:
                outputs = model(signals, hrvs)
            else:
                outputs = model(signals)
                
            loss = criterion(outputs, labels)   
            loss.backward()                     
            optimizer.step()                    
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        epoch_train_acc = accuracy_score(all_train_labels, all_train_preds)

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for signals, hrvs, labels in val_loader:
                signals, hrvs, labels = signals.to(device), hrvs.to(device), labels.to(device)
                if use_hrv:
                    outputs = model(signals, hrvs)
                else:
                    outputs = model(signals)
                
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()
                
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
        scheduler.step()
        
        # Save training logs
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_f1'].append(epoch_train_f1)
        history['val_f1'].append(epoch_val_f1)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"[{epoch+1:03d}]  | "
              f"{epoch_train_loss:.4f}     | "
              f"{epoch_val_loss:.4f}     | "
              f"{epoch_train_f1:.4f}   | "
              f"{epoch_val_f1:.4f}   | "
              f"{epoch_train_acc:.4f}   | "
              f"{epoch_val_acc:.4f}")

        # Checkpoint best model (based on Validation F1)
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            counter = 0
            best_report = classification_report(all_val_labels, all_val_preds)
            best_cm = confusion_matrix(all_val_labels, all_val_preds)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("-" * 85)
    print(f"Training complete! Best Validation Macro F1: {best_val_f1:.4f} (Acc: {best_val_acc:.4f})")
    print(f"Outputs saved in: {run_dir}")
    
    # Save classification report
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(best_report)
        
    # Save history as JSON
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # Save final config with performance summary
    config["best_val_f1"] = best_val_f1
    config["best_val_accuracy"] = best_val_acc
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Save training curves and confusion matrix plots using utils
    plot_training_curves(history, save_path=os.path.join(run_dir, "training_history.png"))
    plot_confusion_matrix(best_cm, save_path=os.path.join(run_dir, "confusion_matrix.png"), title=f'Confusion Matrix: {run_name}')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a 1D ECG model for Paroxysmal Atrial Fibrillation (PAF) prediction.")
    parser.add_argument("--model_type", type=str, default="resnet1d", choices=["resnet1d", "transformer", "senet"],
                        help="Model architecture to train (default: resnet1d)")
    parser.add_argument("--window_seconds", type=int, default=10,
                        help="Input window size in seconds (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs to train (default: 30)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Patience for early stopping (default: 15)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional custom name for the run folder")
    parser.add_argument("--metadata_path", type=str, default="metadata.csv",
                        help="Path to preprocessed metadata.csv")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                        help="Directory containing preprocessed NumPy segment files")
    parser.add_argument("--augment", action="store_true",
                        help="Enable on-the-fly data augmentation (Gaussian noise, baseline wander, scaling) during training")
    parser.add_argument("--weight_scheme", type=str, default="inverse", choices=["none", "inverse", "sqrt"],
                        help="Class weighting scheme for CrossEntropy loss (none, inverse frequency, or square root of frequency)")
    parser.add_argument("--use_sampler", action="store_true",
                        help="Use WeightedRandomSampler to balance batch classes on-the-fly")
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"],
                        help="Loss function type (ce: CrossEntropy, focal: FocalLoss)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma exponent parameter for FocalLoss")
    parser.add_argument("--use_hrv", action="store_true",
                        help="Use 9 hand-crafted HRV features extracted from R-peaks combined with ECG model")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to previous results run directory to load pretrained weights and architecture settings")
    
    args = parser.parse_args()
    
    train(
        model_type=args.model_type,
        window_seconds=args.window_seconds,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        patience=args.patience,
        run_name=args.run_name,
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        augment=args.augment,
        weight_scheme=args.weight_scheme,
        use_sampler=args.use_sampler,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        use_hrv=args.use_hrv,
        resume_from=args.resume_from
    )




