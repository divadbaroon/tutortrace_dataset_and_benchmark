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
    1. Next behavioral state (window-level, 5-class)
       1a. Thinking subtype (4-class, conditional)
       1b. Next behavioral sequence (k=3, k=5)
    2. Error imminence (window-level, binary at multiple horizons)
    3. Query imminence (window-level, binary at multiple horizons)
    4. Post-query improvement (query-level, binary)
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
    print("  NOTE: torch not installed. Sequential/Ensemble baselines will be skipped.")


# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'manifest.yaml')

STATE_NAMES = ['thinking', 'implementing', 'debugging', 'seekingHelp', 'testing']

# Window-level feature groups for ablation
LAYER_1_FEATURES = [
    'code_events', 'terminal_runs', 'terminal_errors', 'test_results',
    'query_count', 'event_density', 'longest_idle_s', 'thinking_time_s',
]

LAYER_2_FEATURES = LAYER_1_FEATURES + [
    'cum_code_rate', 'cum_query_rate', 'query_count_so_far',
    'time_since_session_start_s', 'net_code_growth', 'delete_ratio',
    'time_since_last_query_s', 'error_self_fix',
    'prior_no_effort_rate',
]

LAYER_3_FEATURES = LAYER_2_FEATURES + [
    'segments_in_window', 'pct_thinking', 'pct_implementing',
    'pct_debugging', 'pct_seekingHelp', 'pct_testing',
]

# Query-level feature groups (pre-query behavioral features only)
Q_LAYER_1 = [
    'pre_code_edits', 'pre_terminal_runs', 'pre_terminal_errors',
    'pre_chars_inserted', 'pre_chars_deleted', 'pre_net_code_growth',
    'thinking_time_s', 'pre_duration_s', 'is_first_query',
    'time_since_session_start_s', 'query_index', 'total_queries',
    'pre_time_in_editor_s', 'pre_time_in_terminal_s', 'pre_time_in_chat_s',
]

Q_LAYER_2 = Q_LAYER_1 + [
    'pre_code_edit_rate', 'pre_code_deletes', 'pre_delete_type_ratio',
    'pre_max_consecutive_errors', 'pre_mean_time_between_runs_s',
    'pre_error_self_fix', 'pre_error_ai_fix',
    'pre_error_reading_time_s', 'pre_error_to_edit_s',
    'pre_failed_test_self_fix', 'pre_failed_test_ai_fix',
    'pre_failed_test_to_edit_s',
    'pre_longest_idle_s', 'pre_time_in_task_s', 'pre_time_in_tests_s',
    'pre_response_reading_time_s', 'pre_chat_to_code_latency_s',
    'pre_tab_switches', 'pre_tab_hidden_time_s',
    'thinking_task_s', 'thinking_llm_s', 'thinking_error_s', 'thinking_code_s',
    'time_since_last_query_s',
    'query_length_chars', 'ai_response_length_chars',
]

Q_LAYER_3 = Q_LAYER_2 + [
    'implementing_time_s', 'debugging_time_s', 'testing_time_s',
    'seeking_help_time_s',
]


# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_manifest():
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_dataset(deployment_name, dataset_type):
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
        all_labels = sorted(set(y_test.unique()) | set(y_pred))
        target_names = [STATE_NAMES[i] if i < len(STATE_NAMES) else str(i) for i in all_labels]
        results['per_class'] = classification_report(
            y_test, y_pred, labels=all_labels, target_names=target_names,
            output_dict=True, zero_division=0,
        )

    return results


# ══════════════════════════════════════════════════════════════
#  MLP MODEL
# ══════════════════════════════════════════════════════════════

if HAS_TORCH:
    class MLPModel(nn.Module):
        def __init__(self, input_dim, hidden=128, n_classes=2, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class FlatDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def train_mlp(X_train, y_train, X_test, y_test,
                   n_classes=2, epochs=50, hidden=128, return_probs=False):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train.values if hasattr(X_train, 'values') else X_train)
        Xte = scaler.transform(X_test.values if hasattr(X_test, 'values') else X_test)

        train_dl = DataLoader(FlatDataset(Xtr, y_train.values), batch_size=64, shuffle=True)
        test_dl = DataLoader(FlatDataset(Xte, y_test.values), batch_size=64, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLPModel(Xtr.shape[1], hidden=hidden, n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for bx, by in train_dl:
                bx, by = bx.to(device), by.to(device)
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

        if return_probs:
            return results, probs
        return results


# ══════════════════════════════════════════════════════════════
#  SEQUENCE MODELS
# ══════════════════════════════════════════════════════════════

STATE_TO_IDX = {s: i for i, s in enumerate(STATE_NAMES)}
SUBTYPE_TO_IDX = {
    '': 0, 'none': 0,
    'thinking-task': 1, 'thinking-llm': 2,
    'thinking-error': 3, 'thinking-code': 4,
}
SEG_SEQ_LEN = 30

if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=64, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden=64, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.drop(out[:, -1, :]))

    class TemporalCNNModel(nn.Module):
        def __init__(self, input_dim, hidden=64, n_classes=2, dropout=0.3):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(self.drop(x))

    class TransformerModel(nn.Module):
        def __init__(self, input_dim, hidden=64, n_heads=4, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden)
            self.pos_enc = nn.Parameter(torch.randn(1, SEG_SEQ_LEN, hidden) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            x = self.proj(x) + self.pos_enc[:, :x.size(1), :]
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.fc(self.drop(x))

    class SegSeqDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

    def build_segment_sequences(df, seg_df, time_col='window_end_s', mode='window'):
        feat_dim = 12
        sequences = []

        seg_grouped = {}
        for sid, group in seg_df.groupby('student_id'):
            seg_grouped[str(sid)] = group.sort_values('start_time_ms')

        col = 'window_end_s' if mode == 'window' else 'time_since_session_start_s'

        for _, row in df.iterrows():
            sid = str(row['student_id'])
            cutoff_ms = row[col] * 1000

            student_segs = seg_grouped.get(sid, pd.DataFrame())
            if len(student_segs) > 0:
                student_segs = student_segs[student_segs['start_time_ms'] < cutoff_ms]

            student_segs = student_segs.tail(SEG_SEQ_LEN)

            seq = np.zeros((SEG_SEQ_LEN, feat_dim))
            offset = SEG_SEQ_LEN - len(student_segs)

            for i, (_, seg) in enumerate(student_segs.iterrows()):
                pos = offset + i
                state_idx = STATE_TO_IDX.get(seg['behavioral_state'], 0)
                seq[pos, state_idx] = 1.0

                subtype = seg.get('thinking_subtype', '') or ''
                subtype_idx = SUBTYPE_TO_IDX.get(subtype, 0)
                seq[pos, 5 + subtype_idx] = 1.0

                dur = max(0.01, seg.get('duration_s', 0) or 0)
                seq[pos, 10] = min(dur / 60.0, 5.0)
                seq[pos, 11] = np.log1p(dur)

            sequences.append(seq)

        return sequences

    def train_seq_model(model_class, sequences_train, y_train, sequences_test, y_test,
                        n_classes=2, epochs=50, hidden=64, return_probs=False):
        train_dl = DataLoader(SegSeqDataset(sequences_train, y_train.values), batch_size=64, shuffle=True)
        test_dl = DataLoader(SegSeqDataset(sequences_test, y_test.values), batch_size=64, shuffle=False)

        feat_dim = sequences_train[0].shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(feat_dim, hidden=hidden, n_classes=n_classes).to(device)
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

        if return_probs:
            return results, probs
        return results


def evaluate_ensemble(xgb_probs, seq_probs, y_test, task_type='binary', weight_xgb=0.5):
    avg_probs = weight_xgb * xgb_probs + (1 - weight_xgb) * seq_probs
    preds = avg_probs.argmax(axis=1)

    results = {
        'accuracy': accuracy_score(y_test, preds),
        'macro_f1': f1_score(y_test, preds, average='macro', zero_division=0),
    }
    try:
        if task_type == 'binary':
            results['auc'] = roc_auc_score(y_test, avg_probs[:, 1])
        else:
            results['auc'] = roc_auc_score(y_test, avg_probs, multi_class='ovr', average='macro')
    except:
        results['auc'] = 0.5

    return results


# ══════════════════════════════════════════════════════════════
#  ABLATION RUNNER
# ══════════════════════════════════════════════════════════════

def run_ablation(df_train, df_test, task_name, target_col, feature_layers,
                 task_type='binary', seg_train=None, seg_test=None, time_col='window_end_s'):
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

    print(f"\n  {'Model':<25s}", end='')
    for layer_name in feature_layers:
        print(f"  {layer_name:>22s}", end='')
    print()
    print(f"  {'-' * (25 + 24 * len(feature_layers))}")

    baselines = get_baselines()
    all_results = {}
    layer_names = list(feature_layers.keys())
    last_layer = layer_names[-1]

    xgb_probs = None
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

            if model_name == 'XGBoost' and layer_name == last_layer:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                Xte = scaler.transform(X_test)
                m = clone(model)
                m.fit(Xtr, y_train)
                xgb_probs = m.predict_proba(Xte)
        print()

    if HAS_TORCH:
        print(f"  {'MLP':<25s}", end='')
        all_results['MLP'] = {}
        for layer_name, cols in feature_layers.items():
            avail = [c for c in cols if c in train.columns]
            X_train_layer = train[avail].fillna(0)
            X_test_layer = test[avail].fillna(0)
            res = train_mlp(X_train_layer, y_train, X_test_layer, y_test, n_classes=n_classes)
            all_results['MLP'][layer_name] = res
            print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
        print()

    best_seq_probs = None
    best_seq_auc = 0
    best_seq_name = None

    if HAS_TORCH and seg_train is not None and seg_test is not None:
        mode = 'window' if time_col == 'window_end_s' else 'query'
        seq_tr = build_segment_sequences(train, seg_train, time_col=time_col, mode=mode)
        seq_te = build_segment_sequences(test, seg_test, time_col=time_col, mode=mode)

        seq_models = {
            'Seq-LSTM':        LSTMModel,
            'Seq-GRU':         GRUModel,
            'Seq-CNN':         TemporalCNNModel,
            'Seq-Transformer': TransformerModel,
        }

        for model_name, model_class in seq_models.items():
            print(f"  {model_name:<25s}", end='')
            all_results[model_name] = {}

            for layer_name in layer_names:
                if layer_name == last_layer:
                    res, probs = train_seq_model(
                        model_class, seq_tr, y_train, seq_te, y_test,
                        n_classes=n_classes, return_probs=True
                    )
                    all_results[model_name][layer_name] = res
                    print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')

                    if res.get('auc', 0) > best_seq_auc:
                        best_seq_auc = res['auc']
                        best_seq_probs = probs
                        best_seq_name = model_name
                else:
                    all_results[model_name][layer_name] = {'auc': 0, 'macro_f1': 0, 'accuracy': 0}
                    print(f"  {'—':>14s}", end='')
            print()

    if xgb_probs is not None and best_seq_probs is not None:
        ensemble_name = f'XGB + {best_seq_name}'
        print(f"  {ensemble_name:<25s}", end='')
        all_results[ensemble_name] = {}

        for layer_name in layer_names:
            if layer_name == last_layer:
                res = evaluate_ensemble(xgb_probs, best_seq_probs, y_test, task_type=task_type)
                all_results[ensemble_name][layer_name] = res
                print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
            else:
                all_results[ensemble_name][layer_name] = {'auc': 0, 'macro_f1': 0, 'accuracy': 0}
                print(f"  {'—':>14s}", end='')
        print()

    return all_results


# ══════════════════════════════════════════════════════════════
#  SEQUENCE PREDICTION
# ══════════════════════════════════════════════════════════════

def run_sequence_prediction(df_train, df_test, k):
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

    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

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
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TUTORTRACE BENCHMARK")
    print("=" * 60)

    manifest = load_manifest()
    tasks = manifest.get('tasks', {})
    deployments = manifest.get('deployments', {})

    train_names = [n for n, c in deployments.items() if c.get('split') == 'train' and c.get('enabled', True)]
    test_names = [n for n, c in deployments.items() if c.get('split') == 'test' and c.get('enabled', True)]

    if not train_names or not test_names:
        print("  ERROR: Need at least one train and one test deployment.")
        return

    print(f"\n  Loading data...")
    train_windows = pd.concat([load_dataset(n, 'windows') for n in train_names], ignore_index=True)
    test_windows = pd.concat([load_dataset(n, 'windows') for n in test_names], ignore_index=True)
    train_segments = pd.concat([load_dataset(n, 'segments') for n in train_names], ignore_index=True)
    test_segments = pd.concat([load_dataset(n, 'segments') for n in test_names], ignore_index=True)

    print(f"  Train: {len(train_windows):,} windows, {train_windows['student_id'].nunique()} students")
    print(f"  Test:  {len(test_windows):,} windows, {test_windows['student_id'].nunique()} students")
    print(f"  Segments: {len(train_segments):,} train, {len(test_segments):,} test")

    # Dataset stats
    stats_path = os.path.join(ROOT_DIR, 'figures', 'dataset_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            ds_stats = json.load(f)
        combined = ds_stats.get('combined', {})
        print(f"\n  Dataset overview:")
        for key in ['students', 'students_with_ai', 'tasks_completed', 'total_minutes',
                     'total_events', 'events_no_mouse', 'total_code_edits',
                     'total_terminal_runs', 'total_errors', 'total_queries', 'total_segments']:
            val = combined.get(key, '?')
            label = key.replace('_', ' ').replace('total ', '').title()
            print(f"    {label:<20s} {val:,}" if isinstance(val, int) else f"    {label:<20s} {val}")

    window_layers = {
        'Raw telemetry':     [c for c in LAYER_1_FEATURES if c in train_windows.columns],
        '+Observable':       [c for c in LAYER_2_FEATURES if c in train_windows.columns],
        '+Behav. sequences': [c for c in LAYER_3_FEATURES if c in train_windows.columns],
    }

    results = {}

    # ══════════════════════════════════════════════════════════
    #  TASK 1: NEXT BEHAVIORAL STATE
    # ══════════════════════════════════════════════════════════

    if tasks.get('next_behavioral_state'):
        print("\n" + "=" * 60)
        print("  TASK 1: NEXT BEHAVIORAL STATE (5-class)")
        print("=" * 60)

        valid = train_windows.dropna(subset=['label_next_state'])
        dist = valid['label_next_state'].value_counts()
        print(f"\n  State distribution (train):")
        for state, count in dist.items():
            print(f"    {state}: {count} ({count/len(valid)*100:.1f}%)")

        res = run_ablation(
            train_windows, test_windows,
            'Next behavioral state', 'label_next_state',
            window_layers, task_type='multiclass',
            seg_train=train_segments, seg_test=test_segments,
        )
        if res:
            results['next_behavioral_state'] = res

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
    #  TASK 1a: THINKING SUBTYPE
    # ══════════════════════════════════════════════════════════

    if tasks.get('next_behavioral_state') and 'label_next_thinking_subtype' in train_windows.columns:
        print("\n" + "=" * 60)
        print("  TASK 1a: THINKING SUBTYPE (conditional)")
        print("=" * 60)

        train_think = train_windows[train_windows['label_next_thinking_subtype'].notna()].copy()
        test_think = test_windows[test_windows['label_next_thinking_subtype'].notna()].copy()

        if len(train_think) > 0 and len(test_think) > 0:
            dist = train_think['label_next_thinking_subtype'].value_counts()
            print(f"\n  Subtype distribution (train):")
            for subtype, count in dist.items():
                print(f"    {subtype}: {count} ({count/len(train_think)*100:.1f}%)")

            res = run_ablation(
                train_think, test_think,
                'Thinking subtype', 'label_next_thinking_subtype',
                window_layers, task_type='multiclass',
                seg_train=train_segments, seg_test=test_segments,
            )
            if res:
                results['thinking_subtype'] = res

    # ══════════════════════════════════════════════════════════
    #  TASK 1b: NEXT BEHAVIORAL SEQUENCE
    # ══════════════════════════════════════════════════════════

    if tasks.get('next_behavioral_state'):
        print("\n" + "=" * 60)
        print("  TASK 1b: NEXT BEHAVIORAL SEQUENCE")
        print("=" * 60)

        for k in [3, 5]:
            run_sequence_prediction(train_windows, test_windows, k)

    # ══════════════════════════════════════════════════════════
    #  TASK 2: ERROR IMMINENCE
    # ══════════════════════════════════════════════════════════

    if tasks.get('error_imminence'):
        print("\n" + "=" * 60)
        print("  TASK 2: ERROR IMMINENCE")
        print("=" * 60)

        for horizon in [15, 30, 60]:
            label_col = f'label_error_imminence_{horizon}s'
            if label_col not in train_windows.columns:
                continue

            print(f"\n  --- {horizon}s horizon ---")
            res = run_ablation(
                train_windows, test_windows,
                f'Error imminence ({horizon}s)', label_col,
                window_layers, task_type='binary',
                seg_train=train_segments, seg_test=test_segments,
            )
            if res:
                results[f'error_imminence_{horizon}s'] = res

    # ══════════════════════════════════════════════════════════
    #  TASK 3: QUERY IMMINENCE
    # ══════════════════════════════════════════════════════════

    if tasks.get('query_imminence'):
        print("\n" + "=" * 60)
        print("  TASK 3: QUERY IMMINENCE")
        print("=" * 60)

        for horizon in [5, 10, 15, 30, 45, 60]:
            label_col = f'label_query_imminence_{horizon}s'
            if label_col not in train_windows.columns:
                continue

            print(f"\n  --- {horizon}s horizon ---")
            res = run_ablation(
                train_windows, test_windows,
                f'Query imminence ({horizon}s)', label_col,
                window_layers, task_type='binary',
                seg_train=train_segments, seg_test=test_segments,
            )
            if res:
                results[f'query_imminence_{horizon}s'] = res

    # ══════════════════════════════════════════════════════════
    #  TASK 4: POST-QUERY IMPROVEMENT
    # ══════════════════════════════════════════════════════════

    if tasks.get('post_query_improvement'):
        print("\n" + "=" * 60)
        print("  TASK 4: POST-QUERY IMPROVEMENT")
        print("=" * 60)

        train_queries = pd.concat([load_dataset(n, 'queries') for n in train_names], ignore_index=True)
        test_queries = pd.concat([load_dataset(n, 'queries') for n in test_names], ignore_index=True)

        print(f"  Train: {len(train_queries)} queries | Test: {len(test_queries)} queries")

        q_layers = {
            'Raw telemetry':     [c for c in Q_LAYER_1 if c in train_queries.columns],
            '+Observable':       [c for c in Q_LAYER_2 if c in train_queries.columns],
            '+Behav. sequences': [c for c in Q_LAYER_3 if c in train_queries.columns],
        }

        res = run_ablation(
            train_queries, test_queries,
            'Post-query improvement', 'label_post_query_improvement',
            q_layers, task_type='binary',
            seg_train=train_segments, seg_test=test_segments,
            time_col='time_since_session_start_s',
        )
        if res:
            results['post_query_improvement'] = res

    # ══════════════════════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE (Top 15)")
    print("=" * 60)

    if tasks.get('next_behavioral_state'):
        print_importance(train_windows, 'label_next_state', 'Next behavioral state', LAYER_3_FEATURES)

    if tasks.get('error_imminence'):
        print_importance(train_windows, 'label_error_imminence_15s', 'Error imminence (15s)', LAYER_3_FEATURES)

    if tasks.get('query_imminence'):
        print_importance(train_windows, 'label_query_imminence_15s', 'Query imminence (15s)', LAYER_3_FEATURES)

    if tasks.get('post_query_improvement'):
        train_queries = pd.concat([load_dataset(n, 'queries') for n in train_names], ignore_index=True)
        q_feats = [c for c in Q_LAYER_3 if c in train_queries.columns]
        print_importance(train_queries, 'label_post_query_improvement', 'Post-query improvement', q_feats)

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
    1. Next behavioral state (5-class)
       1a. Thinking subtype (conditional)
       1b. Next behavioral sequence (k=3, k=5)
    2. Error imminence (15s, 30s, 60s)
    3. Query imminence (5s, 10s, 15s, 30s, 45s, 60s)
    4. Post-query improvement (binary)

  Ablation: Raw telemetry → +Observable metrics → +Behavioral sequences
  Baselines: Majority, LogReg, RF{', XGBoost' if HAS_XGBOOST else ''}{', MLP, Seq-LSTM, Seq-GRU, Seq-CNN, Seq-Transformer, XGB+Best' if HAS_TORCH else ''}
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