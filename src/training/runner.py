import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

from src.data.dataloader import get_loaders, get_kfold_loaders
from src.models import instantiate_model
from src.training.loss import FocalLoss
from src.training.engine import train_one_epoch, evaluate
from src.utils.metrics import plot_training_curves, plot_confusion_matrix, plot_eval_curves

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
    resume_from: str = None,
    k_fold: int = 1
):
    """
    Orchestrates deep learning training (standard or K-Fold Group Cross-Validation).
    Saves outputs in a structured folder layout under results/<run_name>/.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle resuming from a checkpoint
    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume directory {resume_from} does not exist.")
        config_path = os.path.join(resume_from, "config.json")
        weights_dir = os.path.join(resume_from, "checkpoints")
        
        # Backward compatibility check for older run directories
        if not os.path.exists(weights_dir):
            weights_dir = resume_from
            
        weights_path = os.path.join(weights_dir, "best_model.pth") if k_fold == 1 else os.path.join(weights_dir, "best_model_fold_0.pth")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing config.json or weights in {resume_from}")
            
        with open(config_path, 'r') as f:
            old_config = json.load(f)
            
        # Override architecture settings to match the loaded weights
        model_type = old_config.get("model_type", model_type)
        use_hrv = old_config.get("use_hrv", use_hrv)
        print(f"Resuming training of {model_type.upper()} from {resume_from} (use_hrv: {use_hrv})")
        
    hrv_dim = 9 if use_hrv else 0
    
    # Define and create structured output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        if resume_from is not None:
            base_run_name = os.path.basename(resume_from.rstrip('/\\'))
            run_name = f"{base_run_name}_continued_{timestamp}"
        else:
            run_name = f"{model_type}_win{window_seconds}_lr{lr}_k{k_fold}_{timestamp}"
    
    run_dir = os.path.join("results", run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "plots", "train")
    reports_dir = os.path.join(run_dir, "reports", "train")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Saving structured outputs to: {run_dir}")
    
    # Load loaders
    if k_fold > 1:
        folds_loaders, test_loader, test_subjects = get_kfold_loaders(
            metadata_path=metadata_path,
            data_dir=data_dir,
            batch_size=batch_size,
            window_seconds=window_seconds,
            k_fold=k_fold,
            augment=augment,
            use_sampler=use_sampler
        )
        print(f"Loaded {k_fold} subject-disjoint validation folds.")
    else:
        train_loader, val_loader, test_loader = get_loaders(
            metadata_path=metadata_path,
            data_dir=data_dir,
            batch_size=batch_size,
            window_seconds=window_seconds,
            augment=augment,
            use_sampler=use_sampler
        )
        folds_loaders = [(train_loader, val_loader)]
        test_subjects = test_loader.dataset.metadata['subject'].unique().tolist()
        
    # Class weights calculation
    first_train_loader = folds_loaders[0][0]
    labels = []
    for _, lbl in first_train_loader.dataset.metadata.iterrows():
        labels.append(lbl['label'])
    
    class_counts = np.bincount(labels)
    weight_scheme = weight_scheme.lower()
    
    if use_sampler:
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
    
    # Setup loss criterion
    if loss_type.lower() == 'focal':
        criterion = FocalLoss(alpha=weights_tensor, gamma=focal_gamma)
        print(f"Loss type: Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print("Loss type: Cross Entropy Loss")
        
    # Initialize global histories
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
        
    fold_best_f1s = []
    fold_best_accs = []
    all_best_val_labels = []
    all_best_val_probs = []
    
    best_cm = None
    best_report = ""
    
    # Trackers for k_fold=1 compatibility
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_probs = []
    best_val_labels = []
    
    print(f"Training starting on {device} (Input window: {window_seconds}s)...")
    
    for fold_idx, (train_loader, val_loader) in enumerate(folds_loaders):
        print(f"\n" + "="*50)
        print(f" TRAINING FOLD {fold_idx + 1} / {k_fold}")
        print("="*50 + "\n")
        
        # Instantiate fresh model
        model = instantiate_model(model_type, hrv_dim, device)
        
        # Load weights if resuming
        if resume_from is not None:
            # Check structured checkpoints dir first, fallback to fallback root
            w_path = os.path.join(resume_from, "checkpoints", f"best_model_fold_{fold_idx}.pth")
            if not os.path.exists(w_path):
                w_path = os.path.join(resume_from, f"best_model_fold_{fold_idx}.pth")
            if not os.path.exists(w_path):
                w_path = os.path.join(resume_from, "checkpoints", "best_model.pth")
            if not os.path.exists(w_path):
                w_path = os.path.join(resume_from, "best_model.pth")
                
            model.load_state_dict(torch.load(w_path, map_location=device))
            print(f"Loaded pretrained weights from {w_path} for fold {fold_idx + 1}.")
            
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        fold_best_val_f1 = 0.0
        fold_best_val_acc = 0.0
        fold_best_val_probs = []
        fold_best_val_labels = []
        fold_best_cm = None
        fold_best_report = ""
        counter = 0
        
        fold_history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_acc': [], 'val_acc': []
        }
        
        print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train F1':<9} | {'Val F1':<9} | {'Train Acc':<9} | {'Val Acc':<9}")
        print("-" * 85)
        
        for epoch in range(num_epochs):
            # A. Training Phase
            epoch_train_loss, epoch_train_f1, epoch_train_acc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_hrv=use_hrv
            )
            
            # B. Validation Phase
            epoch_val_loss, epoch_val_f1, epoch_val_acc, all_val_preds, all_val_probs, all_val_labels = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_hrv=use_hrv
            )
            scheduler.step()
            
            # Save fold logs
            fold_history['train_loss'].append(epoch_train_loss)
            fold_history['val_loss'].append(epoch_val_loss)
            fold_history['train_f1'].append(epoch_train_f1)
            fold_history['val_f1'].append(epoch_val_f1)
            fold_history['train_acc'].append(epoch_train_acc)
            fold_history['val_acc'].append(epoch_val_acc)
            
            print(f"[{epoch+1:03d}]  | "
                  f"{epoch_train_loss:.4f}     | "
                  f"{epoch_val_loss:.4f}     | "
                  f"{epoch_train_f1:.4f}   | "
                  f"{epoch_val_f1:.4f}   | "
                  f"{epoch_train_acc:.4f}   | "
                  f"{epoch_val_acc:.4f}")
                  
            # Checkpoint best epoch
            if epoch_val_f1 > fold_best_val_f1:
                fold_best_val_f1 = epoch_val_f1
                fold_best_val_acc = epoch_val_acc
                checkpoint_name = "best_model.pth" if k_fold == 1 else f"best_model_fold_{fold_idx}.pth"
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, checkpoint_name))
                counter = 0
                fold_best_val_probs = all_val_probs.copy()
                fold_best_val_labels = all_val_labels.copy()
                fold_best_report = classification_report(all_val_labels, all_val_preds, target_names=['Normal/Distant', 'Pre-PAF'])
                fold_best_cm = confusion_matrix(all_val_labels, all_val_preds)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered for Fold {fold_idx+1} at epoch {epoch+1}")
                    break
                    
        print(f"Fold {fold_idx+1} Complete! Best Val F1: {fold_best_val_f1:.4f} (Acc: {fold_best_val_acc:.4f})")
        
        # Save fold metrics to global metrics
        if k_fold > 1:
            history['train_loss'].append(fold_history['train_loss'])
            history['val_loss'].append(fold_history['val_loss'])
            history['train_f1'].append(fold_history['train_f1'])
            history['val_f1'].append(fold_history['val_f1'])
            history['train_acc'].append(fold_history['train_acc'])
            history['val_acc'].append(fold_history['val_acc'])
            
            fold_best_f1s.append(fold_best_val_f1)
            fold_best_accs.append(fold_best_val_acc)
            all_best_val_labels.extend(fold_best_val_labels)
            all_best_val_probs.extend(fold_best_val_probs)
        else:
            history = fold_history
            best_val_f1 = fold_best_val_f1
            best_val_acc = fold_best_val_acc
            best_val_probs = fold_best_val_probs
            best_val_labels = fold_best_val_labels
            best_cm = fold_best_cm
            best_report = fold_best_report
            
    print("-" * 85)
    
    # Post-Training Processing & Logging
    if k_fold > 1:
        mean_f1 = np.mean(fold_best_f1s)
        std_f1 = np.std(fold_best_f1s)
        mean_acc = np.mean(fold_best_accs)
        std_acc = np.std(fold_best_accs)
        
        print(f"Group K-Fold Training Complete!")
        print(f"Validation Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"Validation Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        
        best_report = f"Group K-Fold Cross-Validation Summary ({k_fold} Folds)\n"
        best_report += f"===========================================\n"
        best_report += f"Mean Validation Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}\n"
        best_report += f"Mean Validation Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}\n\n"
        for i, (f1, acc) in enumerate(zip(fold_best_f1s, fold_best_accs)):
            best_report += f"Fold {i+1} - Best Val F1: {f1:.4f} | Acc: {acc:.4f}\n"
            
        best_val_f1 = mean_f1
        best_val_acc = mean_acc
        best_val_probs = all_best_val_probs
        best_val_labels = all_best_val_labels
        best_cm = confusion_matrix(all_best_val_labels, [int(p >= 0.5) for p in all_best_val_probs])
    else:
        print(f"Training Complete! Best Validation Macro F1: {best_val_f1:.4f} (Acc: {best_val_acc:.4f})")
        
    # Save training report files
    with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
        f.write(best_report)
        
    with open(os.path.join(reports_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
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
        "k_fold": k_fold,
        "hrv_mean": first_train_loader.dataset.hrv_mean.tolist() if use_hrv else None,
        "hrv_std": first_train_loader.dataset.hrv_std.tolist() if use_hrv else None,
        "test_subjects": test_subjects,
        "best_val_f1": best_val_f1,
        "best_val_accuracy": best_val_acc
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    # Save training history plots under plots/train/
    plot_training_curves(history, save_path=os.path.join(plots_dir, "training_history.png"))
    plot_confusion_matrix(best_cm, save_path=os.path.join(plots_dir, "confusion_matrix.png"), title=f'Confusion Matrix: {run_name}')
    
    if len(best_val_probs) > 0:
        plot_eval_curves(
            y_true=np.array(best_val_labels),
            y_probs=np.array(best_val_probs),
            save_path=os.path.join(plots_dir, 'validation_evaluation_curves.png'),
            title_prefix=f'Validation Set ({run_name})'
        )
    print(f"All training outputs saved successfully to: {run_dir}")
