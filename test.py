import os
import argparse
from src.evaluation.runner import evaluate_on_test_set, run_challenge_test, score_and_rename_run_dir

def get_latest_run_dir(results_dir: str = 'results') -> str:
    """Returns the most recently created run directory under results."""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory '{results_dir}' does not exist. Please train a model first.")
    runs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not runs:
        raise FileNotFoundError(f"No run directories found in '{results_dir}'.")
    return max(runs, key=os.path.getctime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PAF prediction model ensemble on the test set and run challenge prediction.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to the training run directory. If not specified, defaults to the latest run.")
    parser.add_argument("--metadata_path", type=str, default="metadata.csv",
                        help="Path to preprocessed metadata.csv")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                        help="Directory containing preprocessed NumPy segment files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args()
    
    try:
        run_dir = args.run_dir
        if run_dir is None:
            run_dir = get_latest_run_dir()
        
        # 1. Unbiased test set evaluation
        evaluate_on_test_set(
            run_dir=run_dir,
            metadata_path=args.metadata_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        
        # 2. PhysioNet blind challenge test inference
        run_challenge_test(run_dir=run_dir)
        
        # 3. Official challenge scoring & run directory tagging
        score_and_rename_run_dir(run_dir=run_dir)
        
    except Exception as e:
        print(f"Error: {e}")