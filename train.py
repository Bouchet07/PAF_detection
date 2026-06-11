import argparse
from src.training.runner import train

if __name__ == "__main__":
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
    parser.add_argument("--k_fold", type=int, default=1,
                        help="Number of folds for Group K-Fold Cross-Validation (default: 1, standard train/val split)")
    
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
        resume_from=args.resume_from,
        k_fold=args.k_fold
    )
