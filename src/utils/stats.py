import os
import wfdb

def count_stats():
    """
    Scans the data directories and prints counts and AFIB onset transition statistics.
    """
    print("=== Database Statistics ===")
    
    # 1. AFPDB
    afpdb_path = 'data/paf-prediction-challenge-database/'
    if os.path.exists(afpdb_path):
        afpdb_files = [f[:-4] for f in os.listdir(afpdb_path) if f.endswith('.hea')]
        p_records = [f for f in afpdb_files if f.startswith('p') and not f.endswith('c')]
        n_records = [f for f in afpdb_files if f.startswith('n') and not f.endswith('c')]
        print(f"AFPDB: {len(p_records)} pre-PAF, {len(n_records)} normal records")
    else:
        print("AFPDB: Not found")
        
    # Helper to count dynamic record annotations
    def count_dynamic_dataset(base_path, name):
        if not os.path.exists(base_path):
            print(f"{name}: Not found")
            return
            
        hea_files = []
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.endswith('.hea'):
                    hea_files.append(os.path.join(root, f[:-4]))
                    
        total_records = len(hea_files)
        annotated_records = 0
        total_onsets = 0
        
        for r_path in hea_files:
            atr_path = r_path + '.atr'
            if os.path.exists(atr_path):
                annotated_records += 1
                try:
                    ann = wfdb.rdann(r_path, 'atr')
                    # count '(AFIB' rhythm markings
                    notes = [note for note in ann.aux_note if note == '(AFIB']
                    total_onsets += len(notes)
                except Exception:
                    pass
                    
        print(f"{name}: {total_records} records total, {annotated_records} annotated. Found {total_onsets} AFIB onset transitions.")

    # 2. CPSC2021
    count_dynamic_dataset('data/cpsc2021/cpsc2021/1.0.0/', 'CPSC2021')
    
    # 3. LTAFDB
    count_dynamic_dataset('data/ltafdb/1.0.0/', 'LTAFDB')
    
    # 4. SHDB-AF
    count_dynamic_dataset('data/shdb-af/1.0.1/', 'SHDB-AF')

if __name__ == "__main__":
    count_stats()
