import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import KFold

# Import helpers from evaluation runner to process challenge test records
from src.evaluation.runner import load_test_record

def get_subject_split(df, seed=42, train_ratio=0.7, val_ratio=0.15):
    subjects = df['subject'].unique()
    random.seed(seed)
    subjects_list = list(subjects)
    random.shuffle(subjects_list)
    
    total_subjects = len(subjects_list)
    split1 = int(total_subjects * train_ratio)
    split2 = int(total_subjects * (train_ratio + val_ratio))
    
    train_subjects = set(subjects_list[:split1])
    val_subjects = set(subjects_list[split1:split2])
    test_subjects = set(subjects_list[split2:])
    
    cv_subjects = subjects_list[:split2] # Train + Val for K-Fold
    return train_subjects, val_subjects, test_subjects, cv_subjects

def run_group_kfold(df, cv_subjects, hrv_cols, classifier_class, clf_params, k_fold=5, seed=42):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    val_f1_scores = []
    
    for fold, (train_subj_idx, val_subj_idx) in enumerate(kf.split(cv_subjects)):
        fold_train_subjs = set([cv_subjects[i] for i in train_subj_idx])
        fold_val_subjs = set([cv_subjects[i] for i in val_subj_idx])
        
        train_df = df[df['subject'].isin(fold_train_subjs)].reset_index(drop=True)
        val_df = df[df['subject'].isin(fold_val_subjs)].reset_index(drop=True)
        
        X_train = train_df[hrv_cols].values
        y_train = train_df['label'].values
        X_val = val_df[hrv_cols].values
        y_val = val_df['label'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train
        clf = classifier_class(**clf_params)
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        preds = clf.predict(X_val_scaled)
        val_f1_scores.append(f1_score(y_val, preds, average='macro'))
        
    return np.mean(val_f1_scores), np.std(val_f1_scores)

def score_challenge(clf, scaler, hrv_cols, data_path='data/paf-prediction-challenge-database/', answers_path='event-2-answers'):
    if not os.path.exists(data_path) or not os.path.exists(answers_path):
        print("Challenge data or answers not found. Skipping challenge scoring.")
        return 0.0
        
    all_files = os.listdir(data_path)
    test_records = sorted(list(set([f.replace('.hea', '') for f in all_files if f.startswith('t') and f.endswith('.hea')])))
    
    probabilities = {}
    for name in test_records:
        # Load test record but extract raw HRV features (unnormalized)
        _, hrv_norm_dummy = load_test_record(data_path, name, window_seconds=60, hrv_mean=None, hrv_std=None)
        hrv_raw = hrv_norm_dummy.numpy()[0] # Shape (9,)
        
        # Scale using train scaler
        hrv_scaled = scaler.transform(hrv_raw.reshape(1, -1))
        
        # Predict probability of class 1
        prob_class1 = clf.predict_proba(hrv_scaled)[0][1]
        probabilities[name] = float(prob_class1)
        
    # Official event 2 evaluation
    answers = {}
    with open(answers_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 2:
                rec, lbl = parts
                answers[rec] = lbl
                
    raw_score = 0
    group_a_correct = 0
    group_a_total = 0
    
    for i in range(1, 101, 2):
        r1 = f"t{i:02d}"
        r2 = f"t{i+1:02d}"
        
        lbl1 = answers.get(r1, 'N')
        lbl2 = answers.get(r2, 'N')
        
        p1 = probabilities.get(r1, 0.5)
        p2 = probabilities.get(r2, 0.5)
        
        is_group_a = (lbl1 == 'A' or lbl2 == 'A')
        
        if not is_group_a:
            raw_score += 1
        else:
            group_a_total += 1
            if p1 > p2:
                pred1, pred2 = 'A', 'N'
            else:
                pred1, pred2 = 'N', 'A'
                
            if pred1 == lbl1 and pred2 == lbl2:
                group_a_correct += 1
                raw_score += 1
                
    adjusted_score = raw_score - 22
    pct_score = float(round(adjusted_score / 28 * 100))
    return pct_score

def main():
    metadata_path = "metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found.")
        return
        
    df = pd.read_csv(metadata_path)
    hrv_cols = ['mean_rr', 'std_rr', 'rmssd', 'pnn50', 'mean_hr', 'std_hr', 'lf', 'hf', 'lf_hf_ratio']
    
    # Subject-based disjoint split
    train_subjs, val_subjs, test_subjs, cv_subjects = get_subject_split(df)
    
    train_df = df[df['subject'].isin(train_subjs)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_subjs)].reset_index(drop=True)
    cv_df = df[df['subject'].isin(train_subjs | val_subjs)].reset_index(drop=True)
    
    # Standard evaluation sets
    X_train = train_df[hrv_cols].values
    y_train = train_df['label'].values
    X_test = test_df[hrv_cols].values
    y_test = test_df['label'].values
    
    # Scaler fit on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Classifiers to evaluate
    classifiers = {
        "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"}),
        "Random Forest": (RandomForestClassifier, {"n_estimators": 150, "max_depth": 7, "random_state": 42, "class_weight": "balanced"}),
        "Gradient Boosting": (GradientBoostingClassifier, {"n_estimators": 100, "max_depth": 4, "random_state": 42})
    }
    
    results = {}
    print("="*80)
    print("HRV-ONLY PREDICTOR STUDY (NO RAW ECG SIGNAL)")
    print("="*80)
    
    for name, (clf_class, params) in classifiers.items():
        print(f"\nEvaluating classifier: {name}")
        
        # 1. 5-Fold Group CV
        cv_mean, cv_std = run_group_kfold(df, cv_subjects, hrv_cols, clf_class, params)
        print(f"  5-Fold Group CV F1 Macro: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # 2. Holdout Test Set
        clf = clf_class(**params)
        clf.fit(X_train_scaled, y_train)
        preds_test = clf.predict(X_test_scaled)
        
        f1_test = f1_score(y_test, preds_test, average='macro')
        acc_test = accuracy_score(y_test, preds_test)
        print(f"  Holdout Test F1 Macro: {f1_test:.4f}")
        print(f"  Holdout Test Accuracy: {acc_test:.4f}")
        
        # 3. Score on official Challenge Event 2
        challenge_score = score_challenge(clf, scaler, hrv_cols)
        print(f"  PhysioNet Event 2 Challenge Score: {challenge_score:.1f}%")
        
        results[name] = {
            "cv_f1_mean": float(cv_mean),
            "cv_f1_std": float(cv_std),
            "test_f1": float(f1_test),
            "test_accuracy": float(acc_test),
            "challenge_score": float(challenge_score)
        }
        
    os.makedirs("results", exist_ok=True)
    with open("results/hrv_only_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n" + "="*80)
    print("Finished successfully. Saved results to results/hrv_only_metrics.json")
    print("="*80)

if __name__ == "__main__":
    main()
