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
    3. Query imminence (5s, 10s, 15s, 30s, 45s, 60s)
    4. Query with no effort (binary)
    5. High delegation query (binary)
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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

warnings.filterwarnings('ignore')

# Optional imports
try:
    from xgboost import XGBClassifier, XGBRegressor
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
    'post_thinking_llm_s', 'post_thinking_error_s', 'post_thinking_code_s',
    'time_since_last_query_s',
    'query_length_chars', 'ai_response_length_chars',
    'test_passed_at_query', 'test_total_at_query',
]

Q_LAYER_3 = Q_LAYER_2 + [
    'implementing_time_s', 'debugging_time_s', 'testing_time_s',
    'seeking_help_time_s',
    'post_response_implementing_s', 'post_response_debugging_s',
    'post_response_thinking_s', 'post_response_seeking_help_s',
    'post_response_testing_s',
    'post_code_edits', 'post_code_edit_rate',
    'post_terminal_runs', 'post_terminal_errors',
    'post_error_self_fix',
]

# Features that leak into no-effort prediction (direct effort signals only)
# Thinking/cognitive features (post_thinking_*, post_response_thinking_s,
# post_response_seeking_help_s) are KEPT — they measure cognitive engagement
# after the AI response, not effort toward the task.
POST_LEAKY = {
    'post_code_edits', 'post_code_edit_rate',
    'post_terminal_runs', 'post_terminal_errors',
    'post_error_self_fix',
    'post_response_implementing_s',
    'post_response_debugging_s',
    'post_response_testing_s',
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
#  LSTM MODEL (used by Seg-LSTM)
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
#  SEGMENT-SEQUENCE LSTM (proper behavioral sequence model)
# ══════════════════════════════════════════════════════════════

STATE_TO_IDX = {s: i for i, s in enumerate(STATE_NAMES)}
SUBTYPE_TO_IDX = {
    '': 0, 'none': 0,
    'thinking-task': 1, 'thinking-llm': 2,
    'thinking-error': 3, 'thinking-code': 4,
}
SEG_SEQ_LEN = 30  # max segments to look back

if HAS_TORCH:
    class SegSeqDataset(Dataset):
        """Dataset that provides actual behavioral state sequences."""
        def __init__(self, sequences, labels):
            self.sequences = sequences  # list of (seq_len, feat_dim) arrays
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

    def build_segment_sequences(df, seg_df, time_col='window_end_s', mode='window'):
        """Build actual behavioral state sequences aligned to prediction points.

        For each row in df, gathers the student's segments up to that time point
        and encodes each segment as: [state_one_hot(5), subtype_one_hot(5), duration_s, log_duration]

        Returns: list of numpy arrays, each shape (SEG_SEQ_LEN, 12)
        """
        feat_dim = 12  # 5 state + 5 subtype + duration + log_duration
        sequences = []

        # Pre-group segments by student for efficiency
        seg_grouped = {}
        for sid, group in seg_df.groupby('student_id'):
            seg_grouped[str(sid)] = group.sort_values('start_time_ms')

        col = 'window_end_s' if mode == 'window' else 'time_since_session_start_s'

        for _, row in df.iterrows():
            sid = str(row['student_id'])
            cutoff_ms = row[col] * 1000

            # Get this student's segments up to cutoff
            student_segs = seg_grouped.get(sid, pd.DataFrame())
            if len(student_segs) > 0:
                student_segs = student_segs[student_segs['start_time_ms'] < cutoff_ms]

            # Take last SEG_SEQ_LEN segments
            student_segs = student_segs.tail(SEG_SEQ_LEN)

            # Encode each segment
            seq = np.zeros((SEG_SEQ_LEN, feat_dim))
            offset = SEG_SEQ_LEN - len(student_segs)

            for i, (_, seg) in enumerate(student_segs.iterrows()):
                pos = offset + i

                # State one-hot (5 dims)
                state_idx = STATE_TO_IDX.get(seg['behavioral_state'], 0)
                seq[pos, state_idx] = 1.0

                # Subtype one-hot (5 dims)
                subtype = seg.get('thinking_subtype', '') or ''
                subtype_idx = SUBTYPE_TO_IDX.get(subtype, 0)
                seq[pos, 5 + subtype_idx] = 1.0

                # Duration features (2 dims)
                dur = max(0.01, seg.get('duration_s', 0) or 0)
                seq[pos, 10] = min(dur / 60.0, 5.0)  # normalized, capped at 5 min
                seq[pos, 11] = np.log1p(dur)  # log duration

            sequences.append(seq)

        return sequences

    def train_seg_lstm(sequences_train, y_train, sequences_test, y_test,
                       n_classes=2, epochs=50, hidden=64, return_probs=False):
        """Train LSTM on actual behavioral state sequences."""
        train_dl = DataLoader(
            SegSeqDataset(sequences_train, y_train.values),
            batch_size=64, shuffle=True
        )
        test_dl = DataLoader(
            SegSeqDataset(sequences_test, y_test.values),
            batch_size=64, shuffle=False
        )

        feat_dim = sequences_train[0].shape[1]  # 12
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(feat_dim, hidden=hidden, n_classes=n_classes).to(device)
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


def evaluate_ensemble(xgb_probs, lstm_probs, y_test, task_type='binary', weight_xgb=0.5):
    """Evaluate an ensemble of XGBoost + Seq-LSTM by averaging probabilities."""
    avg_probs = weight_xgb * xgb_probs + (1 - weight_xgb) * lstm_probs
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
    layer_names = list(feature_layers.keys())
    last_layer = layer_names[-1]

    # Classical baselines — capture XGBoost probs for ensemble
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

            # Capture XGBoost probs on last layer for ensemble
            if model_name == 'XGBoost' and layer_name == last_layer:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                Xte = scaler.transform(X_test)
                m = clone(model)
                m.fit(Xtr, y_train)
                xgb_probs = m.predict_proba(Xte)
        print()

    # Seq-LSTM (proper behavioral sequence model)
    lstm_probs = None
    if HAS_TORCH and seg_train is not None and seg_test is not None:
        print(f"  {'Seq-LSTM':<25s}", end='')
        all_results['Seq-LSTM'] = {}

        for layer_name in layer_names:
            if layer_name == last_layer:
                mode = 'window' if time_col == 'window_end_s' else 'query'
                seq_tr = build_segment_sequences(train, seg_train, time_col=time_col, mode=mode)
                seq_te = build_segment_sequences(test, seg_test, time_col=time_col, mode=mode)

                res, lstm_probs = train_seg_lstm(
                    seq_tr, y_train, seq_te, y_test,
                    n_classes=n_classes, return_probs=True
                )
                all_results['Seq-LSTM'][layer_name] = res
                print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
            else:
                all_results['Seq-LSTM'][layer_name] = {'auc': 0, 'macro_f1': 0, 'accuracy': 0}
                print(f"  {'—':>14s}", end='')
        print()

    # Ensemble: XGBoost + Seq-LSTM
    if xgb_probs is not None and lstm_probs is not None:
        print(f"  {'XGB + Seq-LSTM':<25s}", end='')
        all_results['XGB + Seq-LSTM'] = {}

        for layer_name in layer_names:
            if layer_name == last_layer:
                res = evaluate_ensemble(xgb_probs, lstm_probs, y_test, task_type=task_type)
                all_results['XGB + Seq-LSTM'][layer_name] = res
                print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
            else:
                all_results['XGB + Seq-LSTM'][layer_name] = {'auc': 0, 'macro_f1': 0, 'accuracy': 0}
                print(f"  {'—':>14s}", end='')
        print()

    return all_results


# ══════════════════════════════════════════════════════════════
#  REGRESSION RUNNER (time-to-next-query)
# ══════════════════════════════════════════════════════════════

def get_regression_baselines():
    baselines = {
        'Mean': DummyRegressor(strategy='mean'),
        'LinearReg': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        baselines['XGBoost'] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
    return baselines


def evaluate_regression(model, X_train, y_train, X_test, y_test):
    """Train regression model and return metrics."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    # Clip predictions to non-negative
    y_pred = np.maximum(y_pred, 0)

    return {
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'r2': round(r2_score(y_test, y_pred), 4),
    }


def run_regression(df_train, df_test, task_name, target_col, feature_layers,
                   seg_train=None, seg_test=None, time_col='window_end_s',
                   max_target=300):
    """Run regression baselines across feature layers."""
    train = df_train.dropna(subset=[target_col]).copy()
    test = df_test.dropna(subset=[target_col]).copy()

    # Cap target to max_target seconds to reduce outlier influence
    train = train[train[target_col] <= max_target]
    test = test[test[target_col] <= max_target]

    y_train = train[target_col].astype(float)
    y_test = test[target_col].astype(float)

    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Target stats — Train: mean={y_train.mean():.1f}s, median={y_train.median():.1f}s | Test: mean={y_test.mean():.1f}s, median={y_test.median():.1f}s")

    # Header
    print(f"\n  {'Model':<25s}", end='')
    for layer_name in feature_layers:
        print(f"  {layer_name:>22s}", end='')
    print()
    print(f"  {'-' * (25 + 24 * len(feature_layers))}")

    baselines = get_regression_baselines()
    all_results = {}
    layer_names = list(feature_layers.keys())
    last_layer = layer_names[-1]

    for model_name, model in baselines.items():
        print(f"  {model_name:<25s}", end='')
        all_results[model_name] = {}
        for layer_name, cols in feature_layers.items():
            avail = [c for c in cols if c in train.columns]
            X_train = train[avail].fillna(0)
            X_test = test[avail].fillna(0)
            res = evaluate_regression(clone(model), X_train, y_train, X_test, y_test)
            all_results[model_name][layer_name] = res
            print(f"  {res['mae']:>6.1f} / {res['r2']:>.3f}", end='')
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

    for h in [5, 10, 15, 30, 45, 60]:
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

    # Load segment datasets for Seq-LSTM
    train_segments = pd.concat([load_dataset(n, 'segments') for n in train_names], ignore_index=True)
    test_segments = pd.concat([load_dataset(n, 'segments') for n in test_names], ignore_index=True)

    print(f"  Train: {len(train_windows):,} windows, {train_windows['student_id'].nunique()} students")
    print(f"  Test:  {len(test_windows):,} windows, {test_windows['student_id'].nunique()} students")
    print(f"  Segments: {len(train_segments):,} train, {len(test_segments):,} test")

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
            seg_train=train_segments, seg_test=test_segments,
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
    #  TASK 1a: THINKING SUBTYPE PREDICTION
    # ══════════════════════════════════════════════════════════

    if tasks.get('next_behavioral_state') and 'label_next_thinking_subtype' in train_windows.columns:
        print("\n" + "=" * 60)
        print("  TASK 1a: THINKING SUBTYPE (4-class, conditional)")
        print("=" * 60)

        # Filter to windows where next state is thinking
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
        else:
            print("  SKIPPED (insufficient data)")

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
    #  TASK 2b: TIME TO NEXT QUERY (regression)
    # ══════════════════════════════════════════════════════════

    if tasks.get('query_imminence') and 'label_time_to_next_query_s' in train_windows.columns:
        print("\n" + "=" * 60)
        print("  TASK 2b: TIME TO NEXT QUERY (regression)")
        print("=" * 60)

        # Full regression (all windows, capped at 300s)
        print("\n  --- All windows (capped at 300s) ---")
        reg_results = run_regression(
            train_windows, test_windows,
            'Time to next query (all)', 'label_time_to_next_query_s',
            window_layers, max_target=300,
        )
        if reg_results:
            results['time_to_next_query_all'] = reg_results

        # Filtered regression — only windows within 60s of a query
        print("\n  --- Imminent windows only (within 60s of query) ---")
        train_imminent = train_windows[
            (train_windows['label_time_to_next_query_s'].notna()) &
            (train_windows['label_time_to_next_query_s'] <= 60)
        ].copy()
        test_imminent = test_windows[
            (test_windows['label_time_to_next_query_s'].notna()) &
            (test_windows['label_time_to_next_query_s'] <= 60)
        ].copy()

        reg_results_60 = run_regression(
            train_imminent, test_imminent,
            'Time to next query (≤60s)', 'label_time_to_next_query_s',
            window_layers, max_target=60,
        )
        if reg_results_60:
            results['time_to_next_query_imminent'] = reg_results_60

        # Filtered regression — only windows within 30s of a query
        print("\n  --- Imminent windows only (within 30s of query) ---")
        train_imminent_30 = train_windows[
            (train_windows['label_time_to_next_query_s'].notna()) &
            (train_windows['label_time_to_next_query_s'] <= 30)
        ].copy()
        test_imminent_30 = test_windows[
            (test_windows['label_time_to_next_query_s'].notna()) &
            (test_windows['label_time_to_next_query_s'] <= 30)
        ].copy()

        reg_results_30 = run_regression(
            train_imminent_30, test_imminent_30,
            'Time to next query (≤30s)', 'label_time_to_next_query_s',
            window_layers, max_target=30,
        )
        if reg_results_30:
            results['time_to_next_query_30s'] = reg_results_30

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
            seg_train=train_segments, seg_test=test_segments,
            time_col='time_since_session_start_s',
        )
        if res:
            results['query_with_no_effort'] = res

    # ══════════════════════════════════════════════════════════
    #  TASK 4: HIGH DELEGATION QUERY (binary)
    # ══════════════════════════════════════════════════════════

    if tasks.get('query_type'):
        print("\n" + "=" * 60)
        print("  TASK 4: HIGH DELEGATION QUERY (binary)")
        print("=" * 60)

        # Load query datasets
        train_queries = pd.concat([load_dataset(n, 'queries') for n in train_names], ignore_index=True)
        test_queries = pd.concat([load_dataset(n, 'queries') for n in test_names], ignore_index=True)

        # Load query type labels
        label_dfs = []
        for name in train_names + test_names:
            label_path = os.path.join(DATASET_DIR, 'query_labels', f'{name}_labels.csv')
            if os.path.exists(label_path):
                ldf = pd.read_csv(label_path)
                ldf['student_id'] = ldf['student_id'].astype(str)
                label_dfs.append(ldf)
            else:
                print(f"  WARNING: {label_path} not found")

        if label_dfs:
            all_labels = pd.concat(label_dfs, ignore_index=True)

            # Merge labels into queries
            train_queries = train_queries.merge(
                all_labels[['student_id', 'query_index', 'query_type']],
                on=['student_id', 'query_index'], how='left'
            )
            test_queries = test_queries.merge(
                all_labels[['student_id', 'query_index', 'query_type']],
                on=['student_id', 'query_index'], how='left'
            )

            # Drop Unknown and empty labels
            train_queries = train_queries[
                train_queries['query_type'].notna() &
                (train_queries['query_type'] != '') &
                (train_queries['query_type'] != 'Unknown')
            ].copy()
            test_queries = test_queries[
                test_queries['query_type'].notna() &
                (test_queries['query_type'] != '') &
                (test_queries['query_type'] != 'Unknown')
            ].copy()

            # Binary label: high_delegation vs everything else
            train_queries['label_high_delegation'] = (train_queries['query_type'] == 'high_delegation').astype(int)
            test_queries['label_high_delegation'] = (test_queries['query_type'] == 'high_delegation').astype(int)

            print(f"  Train: {len(train_queries)} queries ({train_queries['label_high_delegation'].mean()*100:.1f}% high delegation)")
            print(f"  Test:  {len(test_queries)} queries ({test_queries['label_high_delegation'].mean()*100:.1f}% high delegation)")

            print(f"\n  Query type distribution (train):")
            for qt, count in train_queries['query_type'].value_counts().items():
                print(f"    {qt}: {count}")

            # Use only pre-query features (no post, no query content)
            QUERY_CONTENT_FEATURES = {'query_length_chars', 'ai_response_length_chars'}
            q_layers = {
                'Raw telemetry':     [c for c in Q_LAYER_1 if c in train_queries.columns and c not in POST_LEAKY and c not in QUERY_CONTENT_FEATURES],
                '+Observable':       [c for c in Q_LAYER_2 if c in train_queries.columns and c not in POST_LEAKY and c not in QUERY_CONTENT_FEATURES],
                '+Behav. sequences': [c for c in Q_LAYER_3 if c in train_queries.columns and c not in POST_LEAKY and c not in QUERY_CONTENT_FEATURES],
            }

            res = run_ablation(
                train_queries, test_queries,
                'High delegation query', 'label_high_delegation',
                q_layers, task_type='binary',
                seg_train=train_segments, seg_test=test_segments,
                time_col='time_since_session_start_s',
            )
            if res:
                results['high_delegation'] = res
        else:
            print("  SKIPPED (no query type labels found)")

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

    if 'label_next_thinking_subtype' in train_windows.columns:
        train_think = train_windows[train_windows['label_next_thinking_subtype'].notna()]
        if len(train_think) > 0:
            print_importance(train_think, 'label_next_thinking_subtype', 'Thinking subtype', LAYER_3_FEATURES)

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
    1. Next behavioral state (5-class)
       1a. Thinking subtype (4-class, conditional)
       1b. Next behavioral sequence (k=3, k=5)
    2. Query imminence (5s, 10s, 15s, 30s, 45s, 60s)
       2b. Time to next query (regression)
    3. Query with no effort (binary)
    4. High delegation query (binary)

  Ablation: Raw telemetry → +Observable metrics → +Behavioral sequences
  Baselines: Majority, LogReg, RF{', XGBoost' if HAS_XGBOOST else ''}{', Seq-LSTM, XGB+Seq-LSTM' if HAS_TORCH else ''}
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