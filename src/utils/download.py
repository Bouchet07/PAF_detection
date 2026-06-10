import os
import sys
import wfdb

class Colors:
    if sys.stdout.isatty():
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        RESET = '\033[0m'
    else:
        GREEN = YELLOW = RED = CYAN = BOLD = UNDERLINE = RESET = ""

c = Colors()

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
]

def create_directory(target_dir):
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        print(f"{c.RED}❌ Error creating directory {target_dir}: {e}{c.RESET}")
        return False
    return True

def download_dataset(dataset):
    name = dataset['name']
    db_dir = dataset['db_dir']
    target_dir = dataset['target_dir']

    if not create_directory(target_dir):
        return

    print("\n" + "="*60)
    print(f"{c.CYAN}{c.BOLD}🚀 Starting Download/Update for: {name}{c.RESET}")
    print(f"   Database:  '{c.YELLOW}{db_dir}{c.RESET}'")
    print(f"   Saving to: '{target_dir}'")
    print(f"   Using:     {c.CYAN}wfdb.io.dl_database{c.RESET}")
    print("="*60 + "\n")

    try:
        wfdb.io.dl_database(
            db_dir=db_dir,
            dl_dir=target_dir,
            records='all',
            annotators='all',
            keep_subdirs=True,
            overwrite=False
        )
        print(f"\n{c.GREEN}✅ Finished processing: {name}{c.RESET}")
    except Exception as e:
        print(f"\n{c.RED}❌ Error during download for {name}: {e}{c.RESET}")
        print("   Please check your network connection, directory permissions, and if the")
        print(f"   database '{c.YELLOW}{db_dir}{c.RESET}' exists on PhysioNet.")
    print("-" * 60)

def main():
    print(f"{c.CYAN}{c.BOLD}--- Dataset Download Manager (using 'wfdb' package) ---{c.RESET}")
    print("Checking local dataset status...")
    
    uninstalled_datasets = []
    default_indices = []

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
                idx = int(idx_str) - 1
                if 0 <= idx < len(DATASETS):
                    datasets_to_download.append(DATASETS[idx])
                else:
                    print(f"{c.YELLOW}⚠️ Warning: '{idx+1}' is not a valid dataset number. Skipping.{c.RESET}")
        except ValueError:
            print(f"{c.RED}❌ Error: Invalid input '{choice}'. Could not parse numbers.{c.RESET}")
            return

    if not datasets_to_download:
        print(f"{c.YELLOW}No datasets selected for download.{c.RESET}")
        return

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
