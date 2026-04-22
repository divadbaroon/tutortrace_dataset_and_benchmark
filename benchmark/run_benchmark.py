"""
TutorTrace Benchmark
====================
Runs prediction tasks on prepared datasets.

Prerequisites:
    python prepare_data.py    # generates segments, windows, queries

Usage:
    cd benchmark/
    python run_benchmark.py

Tasks:
    1. Next behavioral state (multiclass)
    2. Next behavioral sequence (k=3, k=5)
    3. Query imminence (15s, 30s, 45s, 60s)
    4. Query with no effort (binary)
"""

import sys
import os
import json
import yaml
import numpy as np
import pandas as pd
import warnings
from collections import Counter

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report,
)

warnings.filterwarnings('ignore')

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  NOTE: xgboost not installed. XGBoost baseline will be skipped.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  NOTE: torch not installed. LSTM/Ensemble baselines will be skipped.")


# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'manifest.yaml')

STATE_NAMES = ['thinking', 'implementing', 'debugging', 'seekingHelp', 'testing']

# Feature groups for ablation
LAYER_1_FEATURES = [
    'code_events', 'terminal_runs', 'terminal_errors', 'test_results',
    'query_count', 'event_density', 'longest_idle_s', 'thinking_time_s',
]

LAYER_2_FEATURES = LAYER_1_FEATURES + [
    'cum_code_rate', 'cum_query_rate', 'query_count_so_far',
    'time_since_session_start_s', 'net_code_growth', 'delete_ratio',
    'time_since_last_query_s', 'error_self_fix',
]

LAYER_3_FEATURES = LAYER_2_FEATURES + [
    'segments_in_window', 'pct_thinking', 'pct_implementing',
    'pct_debugging', 'pct_seekingHelp', 'pct_testing',
]

# Query-level feature groups
Q_LAYER_1 = [
    'pre_code_edits', 'pre_terminal_runs', 'pre_terminal_errors',
    'thinking_time_s', 'pre_duration_s', 'is_first_query',
    'time_since_session_start_s',
]

Q_LAYER_2 = Q_LAYER_1 + [
    'pre_code_edit_rate', 'pre_longest_idle_s',
    'pre_max_consecutive_errors', 'pre_error_self_fix', 'pre_error_ai_fix',
    'time_since_last_query_s', 'test_passed_at_query', 'test_total_at_query',
]

Q_LAYER_3 = Q_LAYER_2 + [
    'implementing_time_s', 'debugging_time_s',
    'testing_time_s', 'seeking_help_time_s',
]

# Features that leak into no-effort prediction
POST_LEAKY = {
    'post_code_edits', 'post_terminal_runs', 'post_terminal_errors',
}


# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_manifest():
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_dataset(deployment_name, dataset_type):
    """Load a prepared CSV by deployment name and type (segments/windows/queries)."""
    paths = {
        'segments': os.path.join(DATASET_DIR, 'behavioral_sequences', f'{deployment_name}_segments.csv'),
        'windows':  os.path.join(DATASET_DIR, 'observable_metrics', 'window_level', f'{deployment_name}_windows.csv'),
        'queries':  os.path.join(DATASET_DIR, 'observable_metrics', 'query_level', f'{deployment_name}_queries.csv'),
    }
    path = paths[dataset_type]
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run prepare_data.py first.")
        return None
    df = pd.read_csv(path)
    df['student_id'] = df['student_id'].astype(str)
    return df


# ══════════════════════════════════════════════════════════════
#  BASELINES
# ══════════════════════════════════════════════════════════════

def get_baselines():
    baselines = {
        'Majority': DummyClassifier(strategy='most_frequent'),
        'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        baselines['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0,
        )
    return baselines


def evaluate(model, X_train, y_train, X_test, y_test, task_type='binary'):
    """Train, predict, return metrics."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(Xte)
        try:
            if task_type == 'binary':
                results['auc'] = roc_auc_score(y_test, y_prob[:, 1])
            else:
                results['auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except:
            results['auc'] = 0.5
    else:
        results['auc'] = 0.5

    if task_type == 'multiclass':
        present_states = sorted(y_test.unique())
        target_names = [STATE_NAMES[i] if i < len(STATE_NAMES) else str(i) for i in present_states]
        results['per_class'] = classification_report(
            y_test, y_pred, target_names=target_names,
            output_dict=True, zero_division=0,
        )

    return results


# ══════════════════════════════════════════════════════════════
#  LSTM
# ══════════════════════════════════════════════════════════════

if HAS_TORCH:
    class SeqDataset(Dataset):
        def __init__(self, features, labels, seq_len=20):
            self.features = features
            self.labels = labels
            self.seq_len = seq_len

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            start = max(0, idx - self.seq_len + 1)
            seq = self.features[start:idx+1]
            if len(seq) < self.seq_len:
                pad = np.zeros((self.seq_len - len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])
            return torch.FloatTensor(seq), torch.LongTensor([self.labels[idx]])

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=64, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    def train_lstm(X_train, y_train, X_test, y_test, n_classes=2, epochs=50):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        train_dl = DataLoader(SeqDataset(Xtr, y_train.values, 20), batch_size=64, shuffle=True)
        test_dl = DataLoader(SeqDataset(Xte, y_test.values, 20), batch_size=64, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(Xtr.shape[1], n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for bx, by in train_dl:
                bx, by = bx.to(device), by.squeeze().to(device)
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()

        model.eval()
        probs, preds = [], []
        with torch.no_grad():
            for bx, _ in test_dl:
                out = model(bx.to(device))
                probs.append(torch.softmax(out, dim=1).cpu().numpy())
                preds.append(out.argmax(dim=1).cpu().numpy())

        probs = np.vstack(probs)
        preds = np.concatenate(preds)

        results = {
            'accuracy': accuracy_score(y_test, preds),
            'macro_f1': f1_score(y_test, preds, average='macro', zero_division=0),
        }
        try:
            if n_classes == 2:
                results['auc'] = roc_auc_score(y_test, probs[:, 1])
            else:
                results['auc'] = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
        except:
            results['auc'] = 0.5

        return results


# ══════════════════════════════════════════════════════════════
#  ABLATION RUNNER
# ══════════════════════════════════════════════════════════════

def run_ablation(df_train, df_test, task_name, target_col, feature_layers, task_type='binary'):
    """Run all baselines across feature layers for one task."""
    train = df_train.dropna(subset=[target_col]).copy()
    test = df_test.dropna(subset=[target_col]).copy()

    if task_type == 'multiclass':
        le = LabelEncoder()
        le.fit(pd.concat([train[target_col], test[target_col]]))
        y_train = pd.Series(le.transform(train[target_col]), index=train.index)
        y_test = pd.Series(le.transform(test[target_col]), index=test.index)
        n_classes = len(le.classes_)
    else:
        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)
        n_classes = 2

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"  {task_name}: SKIPPED (insufficient class diversity)")
        return None

    if task_type == 'binary':
        print(f"  Train: {len(y_train)} (pos: {y_train.mean()*100:.1f}%) | Test: {len(y_test)} (pos: {y_test.mean()*100:.1f}%)")
    else:
        print(f"  Train: {len(y_train)} | Test: {len(y_test)} | Classes: {n_classes}")

    # Header
    print(f"\n  {'Model':<25s}", end='')
    for layer_name in feature_layers:
        print(f"  {layer_name:>22s}", end='')
    print()
    print(f"  {'-' * (25 + 24 * len(feature_layers))}")

    baselines = get_baselines()
    all_results = {}

    # Classical baselines
    for model_name, model in baselines.items():
        print(f"  {model_name:<25s}", end='')
        all_results[model_name] = {}
        for layer_name, cols in feature_layers.items():
            avail = [c for c in cols if c in train.columns]
            X_train = train[avail].fillna(0)
            X_test = test[avail].fillna(0)
            res = evaluate(clone(model), X_train, y_train, X_test, y_test, task_type)
            all_results[model_name][layer_name] = res
            print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
        print()

    # LSTM
    if HAS_TORCH:
        print(f"  {'LSTM':<25s}", end='')
        all_results['LSTM'] = {}
        for layer_name, cols in feature_layers.items():
            avail = [c for c in cols if c in train.columns]
            X_train = train[avail].fillna(0)
            X_test = test[avail].fillna(0)
            res = train_lstm(X_train, y_train, X_test, y_test, n_classes=n_classes)
            all_results['LSTM'][layer_name] = res
            print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
        print()

    return all_results


# ══════════════════════════════════════════════════════════════
#  SEQUENCE PREDICTION
# ══════════════════════════════════════════════════════════════

def run_sequence_prediction(df_train, df_test, k):
    """Predict next k behavioral states."""
    col = f'label_next_{k}_states'
    if col not in df_train.columns:
        return

    train = df_train.dropna(subset=[col])
    test = df_test.dropna(subset=[col])

    if len(train) == 0 or len(test) == 0:
        print(f"  k={k}: SKIPPED (no data)")
        return

    le = LabelEncoder()
    all_seqs = pd.concat([train[col], test[col]])
    le.fit(all_seqs)

    feat_cols = [c for c in LAYER_2_FEATURES if c in train.columns]
    X_train = train[feat_cols].fillna(0)
    X_test = test[feat_cols].fillna(0)

    scaler = StandardScaler()
    rf = RandomForestClassifier(300, random_state=42, n_jobs=-1)
    rf.fit(scaler.fit_transform(X_train), le.transform(train[col]))
    y_pred = le.inverse_transform(rf.predict(scaler.transform(X_test)))
    y_true = test[col].tolist()

    # Exact match
    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    # Per-position accuracy
    pos_acc = [0] * k
    for t, p in zip(y_true, y_pred):
        t_states = t.split(',')[:k]
        p_states = p.split(',')[:k]
        for i in range(min(len(t_states), len(p_states))):
            if t_states[i] == p_states[i]:
                pos_acc[i] += 1
    pos_acc = [c / len(y_true) for c in pos_acc]

    print(f"  k={k}: exact match {exact*100:.1f}%")
    for i, acc in enumerate(pos_acc):
        print(f"    Position {i+1}: {acc*100:.1f}%")


# ══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════

def print_importance(df, target_col, task_name, feature_cols, top_n=15):
    """Train RF and print top features."""
    train = df.dropna(subset=[target_col])
    y = train[target_col]
    if isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
    else:
        y = y.astype(int)

    if y.nunique() < 2:
        return

    avail = [c for c in feature_cols if c in train.columns]
    X = train[avail].fillna(0)

    rf = RandomForestClassifier(300, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=avail).sort_values(ascending=False)

    print(f"\n  {task_name}:")
    print(f"  {'-' * 50}")
    for i, (feat, val) in enumerate(imp.head(top_n).items()):
        print(f"  {i+1:>2d}. {feat:<35s}  {val:.4f}")


# ══════════════════════════════════════════════════════════════
#  WINDOW SIZE ANALYSIS
# ══════════════════════════════════════════════════════════════

def run_window_size_analysis(train_windows, test_windows):
    """Compare query imminence prediction across window sizes (approximate)."""
    print("\n" + "=" * 60)
    print("  WINDOW SIZE ANALYSIS")
    print("=" * 60)
    print("  (Using 30s windows — full analysis requires regenerating windows at each size)")

    feat_cols = [c for c in LAYER_2_FEATURES if c in train_windows.columns]

    print(f"\n  {'Horizon':>10s}  {'AUC':>8s}  {'F1':>8s}  {'Pos%':>8s}")
    print(f"  {'-' * 40}")

    for h in [15, 30, 45, 60]:
        col = f'label_query_imminence_{h}s'
        if col not in train_windows.columns:
            continue

        y_train = train_windows[col].astype(int)
        y_test = test_windows[col].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        X_train = train_windows[feat_cols].fillna(0)
        X_test = test_windows[feat_cols].fillna(0)

        scaler = StandardScaler()
        rf = RandomForestClassifier(300, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(scaler.fit_transform(X_train), y_train)
        y_prob = rf.predict_proba(scaler.transform(X_test))[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, rf.predict(scaler.transform(X_test)), average='macro')
        pos = y_test.mean()

        print(f"  {h:>8d}s  {auc:>8.3f}  {f1:>8.3f}  {pos*100:>7.1f}%")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TUTORTRACE BENCHMARK")
    print("=" * 60)

    # Load manifest
    manifest = load_manifest()
    tasks = manifest.get('tasks', {})
    deployments = manifest.get('deployments', {})

    # Find train/test deployments
    train_names = [n for n, c in deployments.items() if c.get('split') == 'train' and c.get('enabled', True)]
    test_names = [n for n, c in deployments.items() if c.get('split') == 'test' and c.get('enabled', True)]

    if not train_names or not test_names:
        print("  ERROR: Need at least one train and one test deployment.")
        return

    # Load window datasets
    print(f"\n  Loading data...")
    train_windows = pd.concat([load_dataset(n, 'windows') for n in train_names], ignore_index=True)
    test_windows = pd.concat([load_dataset(n, 'windows') for n in test_names], ignore_index=True)

    print(f"  Train: {len(train_windows):,} windows, {train_windows['student_id'].nunique()} students")
    print(f"  Test:  {len(test_windows):,} windows, {test_windows['student_id'].nunique()} students")

    # Window-level feature layers
    window_layers = {
        'Raw telemetry':        [c for c in LAYER_1_FEATURES if c in train_windows.columns],
        '+Observable':          [c for c in LAYER_2_FEATURES if c in train_windows.columns],
        '+Behav. sequences':    [c for c in LAYER_3_FEATURES if c in train_windows.columns],
    }

    results = {}

    # ══════════════════════════════════════════════════════════
    #  TASK 1: NEXT BEHAVIORAL STATE
    # ══════════════════════════════════════════════════════════

    if tasks.get('next_behavioral_state'):
        print("\n" + "=" * 60)
        print("  TASK 1: NEXT BEHAVIORAL STATE (multiclass)")
        print("=" * 60)

        # State distribution
        valid = train_windows.dropna(subset=['label_next_state'])
        dist = valid['label_next_state'].value_counts()
        print(f"\n  State distribution (train):")
        for state, count in dist.items():
            print(f"    {state}: {count} ({count/len(valid)*100:.1f}%)")

        res = run_ablation(
            train_windows, test_windows,
            'Next behavioral state', 'label_next_state',
            window_layers, task_type='multiclass',
        )
        if res:
            results['next_behavioral_state'] = res

            # Per-class breakdown for best model
            best = 'XGBoost' if 'XGBoost' in res else 'RandomForest'
            best_cond = '+Behav. sequences'
            if best in res and best_cond in res[best]:
                pc = res[best][best_cond].get('per_class')
                if pc:
                    print(f"\n  Per-class ({best}, {best_cond}):")
                    print(f"  {'State':<20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'N':>8s}")
                    print(f"  {'-' * 50}")
                    for sn in STATE_NAMES:
                        if sn in pc:
                            s = pc[sn]
                            print(f"  {sn:<20s} {s['precision']:>8.3f} {s['recall']:>8.3f} {s['f1-score']:>8.3f} {int(s['support']):>8d}")

    # ══════════════════════════════════════════════════════════
    #  TASK 1b: NEXT BEHAVIORAL SEQUENCE
    # ══════════════════════════════════════════════════════════

    if tasks.get('next_behavioral_sequence'):
        print("\n" + "=" * 60)
        print("  TASK 1b: NEXT BEHAVIORAL SEQUENCE")
        print("=" * 60)

        for k in [3, 5]:
            run_sequence_prediction(train_windows, test_windows, k)

    # ══════════════════════════════════════════════════════════
    #  TASK 2: QUERY IMMINENCE
    # ══════════════════════════════════════════════════════════

    if tasks.get('query_imminence'):
        print("\n" + "=" * 60)
        print("  TASK 2: QUERY IMMINENCE")
        print("=" * 60)

        for horizon in [15, 30, 45, 60]:
            label_col = f'label_query_imminence_{horizon}s'
            if label_col not in train_windows.columns:
                continue

            print(f"\n  --- {horizon}s horizon ---")
            res = run_ablation(
                train_windows, test_windows,
                f'Query imminence ({horizon}s)', label_col,
                window_layers, task_type='binary',
            )
            if res:
                results[f'query_imminence_{horizon}s'] = res

    # ══════════════════════════════════════════════════════════
    #  TASK 3: QUERY WITH NO EFFORT
    # ══════════════════════════════════════════════════════════

    if tasks.get('query_with_no_effort'):
        print("\n" + "=" * 60)
        print("  TASK 3: QUERY WITH NO EFFORT")
        print("=" * 60)

        # Load query datasets
        train_queries = pd.concat([load_dataset(n, 'queries') for n in train_names], ignore_index=True)
        test_queries = pd.concat([load_dataset(n, 'queries') for n in test_names], ignore_index=True)

        print(f"  Train: {len(train_queries)} queries | Test: {len(test_queries)} queries")

        # Query-level feature layers (strip post features to avoid leakage)
        q_layers = {
            'Raw telemetry':     [c for c in Q_LAYER_1 if c in train_queries.columns and c not in POST_LEAKY],
            '+Observable':       [c for c in Q_LAYER_2 if c in train_queries.columns and c not in POST_LEAKY],
            '+Behav. sequences': [c for c in Q_LAYER_3 if c in train_queries.columns and c not in POST_LEAKY],
        }

        res = run_ablation(
            train_queries, test_queries,
            'Query with no effort', 'label_query_no_effort',
            q_layers, task_type='binary',
        )
        if res:
            results['query_with_no_effort'] = res

    # ══════════════════════════════════════════════════════════
    #  WINDOW SIZE ANALYSIS
    # ══════════════════════════════════════════════════════════

    run_window_size_analysis(train_windows, test_windows)

    # ══════════════════════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE (Top 15)")
    print("=" * 60)

    print_importance(train_windows, 'label_next_state', 'Next behavioral state', LAYER_3_FEATURES)

    if tasks.get('query_with_no_effort'):
        train_queries = pd.concat([load_dataset(n, 'queries') for n in train_names], ignore_index=True)
        q_feats = [c for c in Q_LAYER_3 if c not in POST_LEAKY]
        print_importance(train_queries, 'label_query_no_effort', 'Query with no effort', q_feats)

    # ══════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(f"""
  Dataset:
    Train: {', '.join(train_names)} ({train_windows['student_id'].nunique()} students)
    Test:  {', '.join(test_names)} ({test_windows['student_id'].nunique()} students)

  Tasks:
    1. Next behavioral state (multiclass)
    2. Next behavioral sequence (k=3, k=5)
    3. Query imminence (15s, 30s, 45s, 60s)
    4. Query with no effort (binary)

  Ablation: Raw telemetry → +Observable metrics → +Behavioral sequences
  Baselines: Majority, LogReg, RF{', XGBoost' if HAS_XGBOOST else ''}{', LSTM' if HAS_TORCH else ''}
  Metrics: AUC / Macro F1
    """)

    # Save results
    out_path = os.path.join(ROOT_DIR, 'benchmark', 'results.json')
    serialized = {}
    for task_name, task_res in results.items():
        serialized[task_name] = {}
        for model_name, model_res in task_res.items():
            serialized[task_name][model_name] = {}
            for layer_name, metrics in model_res.items():
                serialized[task_name][model_name][layer_name] = {
                    k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items() if k != 'per_class'
                }

    with open(out_path, 'w') as f:
        json.dump(serialized, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()