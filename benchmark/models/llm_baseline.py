"""
LLM Baseline Evaluation for TutorTrace Benchmark.

Evaluates LLMs on three tasks using the same test data as ML models:
  1. Query imminence (30s horizon) - subsampled 1000 windows
  2. Next behavioral state (5-class) - subsampled 1000 windows
  3. No-effort query detection - full query-level test set

Reads test deployments from manifest.yaml automatically.

Usage:
  cd tutortrace_dataset_and_benchmark
  python3 benchmark/models/llm_baseline.py

Options:
  --root-dir         Path to project root containing manifest.yaml (default: .)
  --sample-size      Number of windows to subsample (default: 1000)
  --model            OpenAI model name (default: gpt-4o-mini)
  --output           Output JSON path (default: llm_results.json)
  --tasks            Comma-separated tasks to run (default: all three)
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from openai import OpenAI


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_imminence_prompt(row):
    return f"""You are analyzing a programming student's IDE activity during a 30-second window. Based on the behavioral features below, predict the probability that this student will submit a query to an AI assistant within the next 30 seconds.

Behavioral features for this window:
- Code edits: {int(row.get('code_events', 0))}
- Terminal runs: {int(row.get('terminal_runs', 0))}
- Terminal errors: {int(row.get('terminal_errors', 0))}
- Event density (events/sec): {row.get('event_density', 0):.2f}
- Longest idle period: {row.get('longest_idle_s', 0):.1f}s
- Thinking time: {row.get('thinking_time_s', 0):.1f}s
- Net code growth (chars): {int(row.get('net_code_growth', 0))}
- Delete ratio: {row.get('delete_ratio', 0):.2f}
- Time since last AI query: {row.get('time_since_last_query_s', 0):.1f}s
- Time since session start: {row.get('time_since_session_start_s', 0):.1f}s
- Cumulative code rate: {row.get('cum_code_rate', 0):.3f}
- Cumulative query rate: {row.get('cum_query_rate', 0):.4f}
- Current behavioral state: {row.get('current_state', 'unknown')}
- Previous behavioral state: {row.get('prev_state', 'unknown')}
- Segments in window: {int(row.get('segments_in_window', 0))}
- % time thinking: {row.get('pct_thinking', 0):.1f}%
- % time implementing: {row.get('pct_implementing', 0):.1f}%
- % time debugging: {row.get('pct_debugging', 0):.1f}%
- % time seeking help: {row.get('pct_seekingHelp', 0):.1f}%
- % time testing: {row.get('pct_testing', 0):.1f}%

Respond with ONLY a number between 0.0 and 1.0 representing the probability the student will query within 30 seconds. No explanation."""


def build_next_state_prompt(row):
    return f"""You are analyzing a programming student's IDE activity during a 30-second window. Based on the behavioral features below, predict what the student will do next.

The possible behavioral states are:
- thinking: pausing to read code, errors, or task description
- implementing: writing new code
- debugging: fixing errors in existing code
- seekingHelp: typing a query to an AI assistant
- testing: running code and reviewing output

Behavioral features for this window:
- Code edits: {int(row.get('code_events', 0))}
- Terminal runs: {int(row.get('terminal_runs', 0))}
- Terminal errors: {int(row.get('terminal_errors', 0))}
- Event density (events/sec): {row.get('event_density', 0):.2f}
- Longest idle period: {row.get('longest_idle_s', 0):.1f}s
- Thinking time: {row.get('thinking_time_s', 0):.1f}s
- Net code growth (chars): {int(row.get('net_code_growth', 0))}
- Delete ratio: {row.get('delete_ratio', 0):.2f}
- Time since last AI query: {row.get('time_since_last_query_s', 0):.1f}s
- Time since session start: {row.get('time_since_session_start_s', 0):.1f}s
- Current behavioral state: {row.get('current_state', 'unknown')}
- Previous behavioral state: {row.get('prev_state', 'unknown')}
- Segments in window: {int(row.get('segments_in_window', 0))}
- % time thinking: {row.get('pct_thinking', 0):.1f}%
- % time implementing: {row.get('pct_implementing', 0):.1f}%
- % time debugging: {row.get('pct_debugging', 0):.1f}%
- % time seeking help: {row.get('pct_seekingHelp', 0):.1f}%
- % time testing: {row.get('pct_testing', 0):.1f}%

Respond with ONLY one of these five words: thinking, implementing, debugging, seekingHelp, testing. No explanation."""


def build_no_effort_prompt(row):
    return f"""You are analyzing a programming student's behavior around an AI query. Based on the features below, predict the probability that after receiving the AI's response, this student will submit another query WITHOUT writing any code or running any terminal commands first (i.e., zero effort between queries).

Pre-query behavior (what the student did before this query):
- Code edits before query: {int(row.get('pre_code_events', row.get('pre_code_edits', 0)))}
- Terminal runs before query: {int(row.get('pre_terminal_runs', 0))}
- Terminal errors before query: {int(row.get('pre_terminal_errors', 0))}
- Thinking time before query: {row.get('pre_thinking_time_s', row.get('thinking_time_s', 0)):.1f}s
- Time in editor before query: {row.get('pre_time_in_editor_s', 0):.1f}s
- Time in chat before query: {row.get('pre_time_in_chat_s', 0):.1f}s
- Longest idle before query: {row.get('pre_longest_idle_s', 0):.1f}s
- Duration before query: {row.get('pre_duration_s', 0):.1f}s

Query characteristics:
- Query length (chars): {int(row.get('query_length_chars', 0))}
- AI response length (chars): {int(row.get('ai_response_length_chars', 0))}

Session context:
- Time since session start: {row.get('time_since_session_start_s', 0):.1f}s
- Time since last query: {row.get('time_since_last_query_s', 0):.1f}s
- Total queries so far: {int(row.get('total_queries', row.get('query_count_so_far', 0)))}

Post-response behavior:
- Time spent reading AI response: {row.get('post_response_thinking_s', 0):.1f}s
- Time thinking about error after response: {row.get('post_thinking_error_s', 0):.1f}s
- Time thinking about code after response: {row.get('post_thinking_code_s', 0):.1f}s

Respond with ONLY a number between 0.0 and 1.0 representing the probability the student will re-query with zero effort. No explanation."""


# ── API caller ───────────────────────────────────────────────────────────────

def call_llm(client, prompt, model="gpt-4o-mini", max_retries=3):
    """Call OpenAI API with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return None


# ── Manifest & data loading ──────────────────────────────────────────────────

def load_manifest(root_dir):
    """Load test deployments from manifest.yaml."""
    manifest_path = os.path.join(root_dir, 'manifest.yaml')
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    test_deployments = []
    for name, config in manifest['deployments'].items():
        if config.get('enabled', True) and config.get('split') == 'test':
            test_deployments.append(name)

    return test_deployments


def load_test_windows(data_dir, test_deployments):
    """Load and concatenate test deployment window CSVs."""
    dfs = []
    for dep in test_deployments:
        path = os.path.join(data_dir, 'observable_metrics', 'window_level', f'{dep}_windows.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['deployment'] = dep
            dfs.append(df)
            print(f"  Loaded {len(df)} windows from {dep}")
        else:
            print(f"  WARNING: {path} not found")
    if not dfs:
        raise FileNotFoundError("No test window files found")
    return pd.concat(dfs, ignore_index=True)


def load_test_queries(data_dir, test_deployments):
    """Load and concatenate test deployment query CSVs."""
    dfs = []
    for dep in test_deployments:
        path = os.path.join(data_dir, 'observable_metrics', 'query_level', f'{dep}_queries.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['deployment'] = dep
            dfs.append(df)
            print(f"  Loaded {len(df)} queries from {dep}")
        else:
            print(f"  WARNING: {path} not found")
    if not dfs:
        raise FileNotFoundError("No test query files found")
    return pd.concat(dfs, ignore_index=True)


def stratified_subsample(df, label_col, n=1000, seed=42):
    """Stratified subsample preserving label distribution."""
    df_clean = df.dropna(subset=[label_col])
    if len(df_clean) <= n:
        print(f"  Dataset size ({len(df_clean)}) <= sample size ({n}), using full set")
        return df_clean

    groups = df_clean.groupby(label_col)
    samples = []
    for label, group in groups:
        frac = len(group) / len(df_clean)
        k = max(1, int(n * frac))
        k = min(k, len(group))
        samples.append(group.sample(n=k, random_state=seed))

    result = pd.concat(samples).sample(frac=1, random_state=seed)
    print(f"  Subsampled {len(result)} from {len(df_clean)} (target: {n})")
    return result


# ── Task runners ─────────────────────────────────────────────────────────────

def run_imminence_task(client, df, model, sample_size):
    """Run query imminence 30s prediction."""
    print("\n" + "=" * 60)
    print("  TASK: QUERY IMMINENCE (30s horizon)")
    print("=" * 60)

    label_col = 'label_query_imminence_30s'
    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found in data")
        return None

    sample = stratified_subsample(df, label_col, n=sample_size)
    pos_rate = sample[label_col].mean()
    print(f"  Positive rate: {pos_rate:.1%}")

    predictions = []
    labels = []
    errors = 0

    for i, (idx, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(sample)}...")

        prompt = build_imminence_prompt(row)
        response = call_llm(client, prompt, model)

        if response is None:
            errors += 1
            continue

        try:
            prob = float(response.strip())
            prob = max(0.0, min(1.0, prob))
            predictions.append(prob)
            labels.append(int(row[label_col]))
        except ValueError:
            errors += 1

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions, skipping")
        return None

    auc = roc_auc_score(labels, predictions)
    binary_preds = [1 if p > 0.5 else 0 for p in predictions]
    f1 = f1_score(labels, binary_preds, average='macro')

    print(f"\n  Results:")
    print(f"  Valid predictions: {len(predictions)}/{len(sample)} ({errors} errors)")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Macro F1: {f1:.3f}")

    return {
        'task': 'query_imminence_30s',
        'auc': round(auc, 3),
        'f1': round(f1, 3),
        'n_samples': len(predictions),
        'n_errors': errors,
        'pos_rate': round(pos_rate, 3),
    }


def run_next_state_task(client, df, model, sample_size):
    """Run next behavioral state prediction."""
    print("\n" + "=" * 60)
    print("  TASK: NEXT BEHAVIORAL STATE (5-class)")
    print("=" * 60)

    label_col = 'label_next_state'
    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found in data")
        return None

    STATE_MAP = {'thinking': 0, 'implementing': 1, 'debugging': 2, 'seekingHelp': 3, 'testing': 4}
    STATE_NAMES = {v: k for k, v in STATE_MAP.items()}

    sample = stratified_subsample(df, label_col, n=sample_size)

    predictions = []
    labels = []
    errors = 0

    for i, (idx, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(sample)}...")

        prompt = build_next_state_prompt(row)
        response = call_llm(client, prompt, model)

        if response is None:
            errors += 1
            continue

        response_clean = response.strip().lower()

        # Try to match response to a valid state
        matched_state = None
        for state_name in STATE_MAP:
            if state_name.lower() in response_clean:
                matched_state = state_name
                break

        if matched_state is None:
            errors += 1
            continue

        predictions.append(STATE_MAP[matched_state])
        label_val = row[label_col]
        if isinstance(label_val, str):
            labels.append(STATE_MAP.get(label_val, -1))
        else:
            labels.append(int(label_val))

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions, skipping")
        return None

    # AUC requires one-hot encoding for multiclass
    from sklearn.preprocessing import label_binarize
    unique_labels = sorted(set(labels) | set(predictions))
    if len(unique_labels) < 2:
        print(f"  ERROR: Only {len(unique_labels)} unique labels, cannot compute AUC")
        return None

    labels_bin = label_binarize(labels, classes=unique_labels)
    preds_bin = label_binarize(predictions, classes=unique_labels)

    try:
        auc = roc_auc_score(labels_bin, preds_bin, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.5

    f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    print(f"\n  Results:")
    print(f"  Valid predictions: {len(predictions)}/{len(sample)} ({errors} errors)")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Macro F1: {f1:.3f}")

    report = classification_report(
        labels, predictions,
        target_names=[STATE_NAMES[i] for i in unique_labels],
        labels=unique_labels,
        zero_division=0,
    )
    print(f"\n{report}")

    return {
        'task': 'next_behavioral_state',
        'auc': round(auc, 3),
        'f1': round(f1, 3),
        'n_samples': len(predictions),
        'n_errors': errors,
    }


def run_no_effort_task(client, df_queries, model):
    """Run no-effort query detection."""
    print("\n" + "=" * 60)
    print("  TASK: QUERY WITH NO EFFORT (binary)")
    print("=" * 60)

    label_col = 'label_query_no_effort'
    if label_col not in df_queries.columns:
        print(f"  ERROR: {label_col} not found in data")
        return None

    df_clean = df_queries.dropna(subset=[label_col])
    pos_rate = df_clean[label_col].mean()
    print(f"  Total queries: {len(df_clean)}")
    print(f"  Positive rate: {pos_rate:.1%}")

    predictions = []
    labels = []
    errors = 0

    for i, (idx, row) in enumerate(df_clean.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(df_clean)}...")

        prompt = build_no_effort_prompt(row)
        response = call_llm(client, prompt, model)

        if response is None:
            errors += 1
            continue

        try:
            prob = float(response.strip())
            prob = max(0.0, min(1.0, prob))
            predictions.append(prob)
            labels.append(int(row[label_col]))
        except ValueError:
            errors += 1

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions, skipping")
        return None

    auc = roc_auc_score(labels, predictions)
    binary_preds = [1 if p > 0.5 else 0 for p in predictions]
    f1 = f1_score(labels, binary_preds, average='macro')

    print(f"\n  Results:")
    print(f"  Valid predictions: {len(predictions)}/{len(df_clean)} ({errors} errors)")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Macro F1: {f1:.3f}")

    return {
        'task': 'query_no_effort',
        'auc': round(auc, 3),
        'f1': round(f1, 3),
        'n_samples': len(predictions),
        'n_errors': errors,
        'pos_rate': round(pos_rate, 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LLM Baseline Evaluation')
    parser.add_argument('--root-dir', default='.', help='Path to project root containing manifest.yaml')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Subsample size for window-level tasks')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model name')
    parser.add_argument('--output', default='llm_results.json', help='Output JSON path')
    parser.add_argument('--tasks', default='imminence,next_state,no_effort',
                        help='Comma-separated tasks to run')
    args = parser.parse_args()

    tasks_to_run = [t.strip() for t in args.tasks.split(',')]
    data_dir = os.path.join(args.root_dir, 'dataset')

    # Load test deployments from manifest
    test_deployments = load_manifest(args.root_dir)
    if not test_deployments:
        print("  ERROR: No enabled test deployments found in manifest.yaml")
        return

    print("=" * 60)
    print("  TUTORTRACE LLM BASELINE")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Test deployments: {test_deployments}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Tasks: {tasks_to_run}")
    print()

    client = OpenAI()
    results = []

    # Load window-level data if needed
    df_windows = None
    if 'imminence' in tasks_to_run or 'next_state' in tasks_to_run:
        print("  Loading test windows...")
        df_windows = load_test_windows(data_dir, test_deployments)
        print(f"  Total test windows: {len(df_windows)}")

    # Load query-level data if needed
    df_queries = None
    if 'no_effort' in tasks_to_run:
        print("\n  Loading test queries...")
        df_queries = load_test_queries(data_dir, test_deployments)
        print(f"  Total test queries: {len(df_queries)}")

    # Run tasks
    if 'imminence' in tasks_to_run and df_windows is not None:
        result = run_imminence_task(client, df_windows, args.model, args.sample_size)
        if result:
            results.append(result)

    if 'next_state' in tasks_to_run and df_windows is not None:
        result = run_next_state_task(client, df_windows, args.model, args.sample_size)
        if result:
            results.append(result)

    if 'no_effort' in tasks_to_run and df_queries is not None:
        result = run_no_effort_task(client, df_queries, args.model)
        if result:
            results.append(result)

    # Save results
    output = {
        'model': args.model,
        'test_deployments': test_deployments,
        'sample_size': args.sample_size,
        'results': results,
    }

    output_path = os.path.join(args.root_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<30s} {'AUC':>8s} {'F1':>8s} {'N':>8s}")
    print(f"  {'-' * 56}")
    for r in results:
        print(f"  {r['task']:<30s} {r['auc']:>8.3f} {r['f1']:>8.3f} {r['n_samples']:>8d}")


if __name__ == '__main__':
    main()