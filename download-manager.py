import os
import sys
import wfdb

# --- ANSI Color Class ---
# We add this class to handle terminal colors
class Colors:
    # Check if the terminal supports colors (a basic check)
    # We'll assume yes if it's a TTY, but disable if it fails
    if sys.stdout.isatty():
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        RESET = '\033[0m'
    else:
        # If not a TTY (e.g., piping to a file), disable colors
        GREEN = YELLOW = RED = CYAN = BOLD = UNDERLINE = RESET = ""

# Create an instance for easy use
c = Colors()


# --- Configuration: Add all your datasets here ---
# We now use 'db_dir' instead of 'url' and 'wget_cut_dirs'

DATASETS = [
    {
        "name": "PAF Prediction Challenge Database (afpdb)",
        "db_dir": "afpdb",
        "target_dir": "data/paf-prediction-challenge-database",
    },
    {
        "name": "SHDB-AF: a Japanese Holter ECG database of atrial fibrillation",
        "db_dir": "shdb-af",
        "target_dir": "data/shdb-af-database",
    },
    {
        "name": "MIT-BIH Arrhythmia Database",
        "db_dir": "afdb",
        "target_dir": "data/mit-bih-arrhythmia-database",
    },
    {
        "name": "The 4th China Physiological Signal Challenge 2021",
        "db_dir": "cpsc2021",
        "target_dir": "data/cpsc2021-challenge-database",
    },
    {
        "name": "Long Term AF Database",
        "db_dir": "ltafdb",
        "target_dir": "data/long-term-af-database",
    }
    # --- Add your next dataset here ---
    # {
    #     "name": "Another Dataset",
    #     "db_dir": "mitdb", # The PhysioNet database path (e.g., 'mitdb')
    #     "target_dir": "data/my-other-dataset-folder",
    # },
]

# --------------------------------------------------
def create_directory(target_dir):
    """Creates the target directory if it doesn't exist."""
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        print(f"{c.RED}❌ Error creating directory {target_dir}: {e}{c.RESET}")
        return False
    return True

# --------------------------------------------------
def download_dataset(dataset):
    """
    Downloads a dataset using the wfdb.io.dl_database function.
    """
    name = dataset['name']
    db_dir = dataset['db_dir']
    target_dir = dataset['target_dir']

    # This is still good practice, though dl_database might handle it.
    if not create_directory(target_dir):
        return

    # --- Print colorful status ---
    print("\n" + "="*60)
    print(f"{c.CYAN}{c.BOLD}🚀 Starting Download/Update for: {name}{c.RESET}")
    print(f"   Database:  '{c.YELLOW}{db_dir}{c.RESET}'")
    print(f"   Saving to: '{target_dir}'")
    print(f"   Using:     {c.CYAN}wfdb.io.dl_database{c.RESET}")
    print("="*60 + "\n")

    try:
        # --- This is the new download command ---
        # It uses the 'wfdb' package to download the entire database
        wfdb.io.dl_database(
            db_dir=db_dir,
            dl_dir=target_dir,
            records='all',      # Download all records listed in the RECORDS file
            annotators='all',   # Download all annotators (if ANNOTATORS file exists)
            keep_subdirs=True,  # Replicate PhysioNet's directory structure
            overwrite=False     # Mimics wget -N -c (skip if same size, resume if partial)
        )
        # --- End of new download command ---
        
        print(f"\n{c.GREEN}✅ Finished processing: {name}{c.RESET}")

    # Catch potential network errors, permissions errors, "database not found", etc.
    except Exception as e:
        print(f"\n{c.RED}❌ Error during download for {name}: {e}{c.RESET}")
        print("   Please check your network connection, directory permissions, and if the")
        print(f"   database '{c.YELLOW}{db_dir}{c.RESET}' exists on PhysioNet.")
    
    print("-" * 60)

# --------------------------------------------------
def main():
    """
    Shows a colorful, native text-based menu.
    (This function remains unchanged as its logic is still valid)
    """
    print(f"{c.CYAN}{c.BOLD}--- Dataset Download Manager (using 'wfdb' package) ---{c.RESET}")
    print("Checking local dataset status...")
    
    uninstalled_datasets = []
    default_indices = []

    # --- Display the menu with colors ---
    print(f"\n{c.BOLD}{c.UNDERLINE}Available Datasets:{c.RESET}")
    for i, ds in enumerate(DATASETS):
        is_installed = os.path.exists(ds['target_dir'])
        
        if is_installed:
            status = f"{c.GREEN}[INSTALLED]{c.RESET}"
        else:
            status = f"{c.YELLOW}[NOT INSTALLED]{c.RESET}"
            uninstalled_datasets.append(ds)
            default_indices.append(str(i+1))
        
        print(f"   {c.BOLD}({i+1}){c.RESET} {status:<28} {ds['name']}")
    
    print(f"\n{c.CYAN}" + "-" * 60 + f"{c.RESET}")
    
    # --- Get User Input ---
    default_prompt = f"'{', '.join(default_indices)}'" if default_indices else "none"
    print(f"{c.BOLD}Enter the numbers of datasets to download (e.g., '1, 2').{c.RESET}")
    print(f"- To download all *not installed* ({c.YELLOW}{default_prompt}{c.RESET}), just press {c.BOLD}ENTER{c.RESET}.")
    print(f"- To download *all* datasets (inc. updates), type '{c.BOLD}all{c.RESET}'.")
    print(f"- To quit, type '{c.BOLD}q{c.RESET}'.")

    try:
        choice = input(f"{c.BOLD}Your choice: {c.RESET}").strip().lower()
    except KeyboardInterrupt:
        print(f"\n{c.YELLOW}Selection cancelled. Exiting.{c.RESET}")
        sys.exit(0)

    datasets_to_download = []

    # --- Process User Input ---
    if choice == 'q':
        print("Quitting.")
        return
        
    elif choice == 'all':
        print(f"{c.GREEN}Selected: ALL datasets.{c.RESET}")
        datasets_to_download = DATASETS
        
    elif choice == '' and uninstalled_datasets:
        print(f"{c.GREEN}Selected: Default (all not installed).{c.RESET}")
        datasets_to_download = uninstalled_datasets
        
    elif choice == '' and not uninstalled_datasets:
        print("No datasets selected (and all are installed). Exiting.")
        return
        
    else:
        print(f"{c.GREEN}Selected: Custom list '{choice}'.{c.RESET}")
        try:
            indices = choice.split(',')
            for idx_str in indices:
                idx_str = idx_str.strip()
                if not idx_str:
                    continue
                
                idx = int(idx_str) - 1 # Convert from 1-based to 0-based
                
                if 0 <= idx < len(DATASETS):
                    datasets_to_download.append(DATASETS[idx])
                else:
                    print(f"{c.YELLOW}⚠️ Warning: '{idx+1}' is not a valid dataset number. Skipping.{c.RESET}")
        except ValueError:
            print(f"{c.RED}❌ Error: Invalid input '{choice}'. Could not parse numbers.{c.RESET}")
            return

    # --- Execute Downloads ---
    if not datasets_to_download:
        print(f"{c.YELLOW}No datasets selected for download.{c.RESET}")
        return

    # De-duplicate list
    unique_datasets = []
    seen_names = set()
    for ds in datasets_to_download:
        if ds['name'] not in seen_names:
            unique_datasets.append(ds)
            seen_names.add(ds['name'])

    for ds in unique_datasets:
        download_dataset(ds)

    print(f"\n{c.GREEN}{c.BOLD}All selected operations are complete.{c.RESET}")


if __name__ == "__main__":
    main()