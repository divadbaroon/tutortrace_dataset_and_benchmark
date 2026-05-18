# A Dataset for Novice Programmer Behavior Prediction in AI-Assisted Coding

## Overview

IDETrace is a large-scale behavioral telemetry dataset capturing fine-grained student interactions with an AI-assisted programming environment. The dataset spans **8 classroom deployments** across introductory Python courses, comprising:

- **664 students (367 using the AI tutor)
- **882,367 telemetry events** (204,092 excluding mouse movements)
- **15,991 behavioral sequences** (auto-segmented into thinking, implementing, debugging, seeking help, and testing states)
- **89,015 observable metric observations** (87,439 window-level + 1,576 query-level)
- **1,692 labeled queries** classified as *guided* or *dependent* help-seeking

The dataset is organized into a three-layer abstraction:

1. **Raw telemetry** — timestamped IDE events (code edits, terminal runs, errors, AI queries/responses, test results)
2. **Observable metrics** — sliding window features (30s window, 5s step) and per-query behavioral features
3. **Behavioral sequences** — auto-segmented behavioral states with thinking subtypes

## Benchmark Tasks

We define four prediction tasks operating on behavioral telemetry:

| Task | Granularity | Type | Description |
|------|-------------|------|-------------|
| **Next Behavioral State** | Window | 5-class | Predict the student's next behavioral state |
| **Error Imminence** | Window | Binary | Predict whether a terminal error will occur within *h* seconds |
| **Query Imminence** | Window | Binary | Predict whether the student will query the AI within *h* seconds |
| **Query Type** | Query (t-15s) | Binary | Predict whether the upcoming query will be *guided* or *dependent* |

Tasks 1–3 operate on 30-second sliding windows. Task 4 operates at the query level with a 15-second anticipation window, aligned with query imminence detection — at the moment the system detects a query is imminent, it simultaneously predicts the query type.

## Repository Structure

```
├── manifest.yaml                          # Dataset configuration
├── prepare_data.py                        # Data preparation pipeline
├── dataset/
│   ├── raw_telemetry/                     # Raw event streams (JSON)
│   │   ├── deployment_1_telemetry.json
│   │   └── ...
│   ├── behavioral_sequences/              # Auto-segmented behavioral states
│   │   ├── deployment_1_segments.csv
│   │   └── ...
│   ├── observable_metrics/
│   │   ├── window_level/                  # 30s sliding window features
│   │   │   ├── deployment_1_windows.csv
│   │   │   └── ...
│   │   └── query_level/                   # Per-query behavioral features
│   │       ├── deployment_1_queries.csv
│   │       └── ...
│   └── query_labels/                      # Guided/dependent labels
│       ├── deployment_1_labels.csv
│       └── ...
├── benchmark/
│   ├── run_benchmark.py                   # Main benchmark runner
│   ├── run_all_benchmark.py               # Cross-deployment evaluation
│   ├── results.json                       # Cached results
│   └── models/
│       ├── llm_baseline.py                # Closed-source LLM evaluation
│       └── open_llm_baseline.py           # Open-source LLM evaluation
├── behavioral_classifier/
│   └── auto_segmenter.py                  # Behavioral state classifier
└── figures/
    └── export_figure_data.py              # Figure data generation
```

## Deployments

| ID | Split | Instructor | Task | Duration | Students | AI Users | Queries |
|----|-------|------------|------|----------|----------|----------|---------|
| D1 | Train | A | Playlist | Full | 190 | 94 | 428 |
| D2 | Test | A | Playlist | Full | 113 | 90 | 536 |
| D3 | Test | B | GradeBook | Full | 49 | 48 | 190 |
| D4 | Test | B | Rectangle | 5 min | 14 | 13 | 47 |
| D5 | Test | B | Rectangle | 5 min | 37 | 36 | 99 |
| D6 | Test | C | Rectangle | Full | 25 | 24 | 120 |
| D7 | Test | A | Rectangle | Full | 43 | 42 | 202 |
| D8 | Test | A | Rectangle | Full | 15 | 14 | 59 |

## Quick Start

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost torch pyyaml
```

### 1. Prepare derived datasets

If starting from raw telemetry:

```bash
python prepare_data.py --force
```

This generates behavioral sequences, window-level features, and query-level features for all enabled deployments.

### 2. Run benchmark (Setup A: D1 → D2)

```bash
cd benchmark
python run_benchmark.py
```

This runs all four tasks with all model baselines (classical ML, MLP, sequence models, ensembles) and saves results to `benchmark/results.json`.

### 3. Run cross-deployment evaluation

```bash
python benchmark/run_all_benchmark.py
```

This evaluates generalization by training on D1 and testing on each other deployment individually (per-deployment) and on combined held-out sets (Setup B).

### 4. Run LLM baselines

Closed-source (requires OpenAI API key):

```bash
python benchmark/models/llm_baseline.py --model gpt-4o-mini
python benchmark/models/llm_baseline.py --model gpt-4o
```

Open-source (requires Ollama):

```bash
python benchmark/models/open_llm_baseline.py --model llama3.1:8b
python benchmark/models/open_llm_baseline.py --model qwen3.5:9b
python benchmark/models/open_llm_baseline.py --model deepseek-r1:8b
```

## Data Format

### Raw Telemetry (JSON)

```json
{
  "student_id": {
    "events": [
      {"timestamp": 1234567890, "type": "CODE_TYPE", "payload": {...}},
      {"timestamp": 1234567891, "type": "TERMINAL_RUN", "payload": {...}},
      {"timestamp": 1234567892, "type": "TERMINAL_ERROR", "payload": {"message": "..."}},
      {"timestamp": 1234567893, "type": "CHAT_QUERY", "payload": {"text": "..."}},
      {"timestamp": 1234567894, "type": "CHAT_RESPONSE", "payload": {"text": "..."}}
    ]
  }
}
```

### Behavioral Sequences (CSV)

| Column | Description |
|--------|-------------|
| `student_id` | Student identifier |
| `segment_index` | Sequential segment number |
| `behavioral_state` | One of: thinking, implementing, debugging, seekingHelp, testing |
| `thinking_subtype` | If thinking: thinking-task, thinking-llm, thinking-error, thinking-code |
| `start_time_ms` | Segment start (ms from session start) |
| `end_time_ms` | Segment end (ms from session start) |
| `duration_s` | Segment duration in seconds |

### Window-Level Features (CSV)

Features organized into three abstraction layers:

**Layer 1 — Raw telemetry:** `code_events`, `terminal_runs`, `terminal_errors`, `test_results`, `query_count`, `event_density`, `longest_idle_s`, `thinking_time_s`

**Layer 2 — Observable metrics:** `cum_code_rate`, `cum_query_rate`, `query_count_so_far`, `time_since_session_start_s`, `net_code_growth`, `delete_ratio`, `time_since_last_query_s`, `error_self_fix`, `prior_no_effort_rate`

**Layer 3 — Behavioral sequences:** `segments_in_window`, `pct_thinking`, `pct_implementing`, `pct_debugging`, `pct_seekingHelp`, `pct_testing`, `current_state`, `prev_state`

**Labels:** `label_next_state`, `label_error_imminence_{5,10,15,30,45,60}s`, `label_query_imminence_{5,10,15,30,45,60}s`, `label_next_query_type`

### Query-Level Features (CSV)

Pre-query behavioral features computed from [last AI response → 15 seconds before query], including code edit patterns, terminal activity, error handling, timing, and behavioral state durations. See `prepare_data.py` for the full feature list.

### Query Labels (CSV)

| Column | Description |
|--------|-------------|
| `student_id` | Student identifier |
| `query_index` | 1-indexed query number within session |
| `query_type` | `guided` or `dependent` |

**Guided:** The student did cognitive work to formulate what they need — asked a specific question, identified a problem, or described what they tried.

**Dependent:** The student did NOT do cognitive work — vague requests, pasted code/instructions with no question, or delegated to the AI.

Labels were generated by GPT-4o and validated via inter-rater reliability with two human annotators (κ = .897 human-human, κ = .709/.690 GPT-human, 97 queries).

## Model Baselines

| Category | Models |
|----------|--------|
| Classical ML | Majority, Logistic Regression, Random Forest (300 trees), XGBoost |
| Neural | 3-layer MLP with dropout |
| Sequential | LSTM, GRU, Temporal CNN, Transformer encoder (last 30 behavioral segments) |
| Ensemble | XGBoost + best sequential model (probability averaging) |
| LLM (zero-shot) | GPT-4o-mini, GPT-4o, GPT-5.5, Llama 3.1 8B, Qwen 3.5 9B, DeepSeek-R1 8B |

## Key Results (Setup A: D1 → D2)

| Task | Best Model | AUC | Macro F1 |
|------|-----------|:---:|:---:|
| Next Behavioral State | XGB+Seq-CNN | .828 | .495 |
| Error Imminence (15s) | XGB+Seq-Trans | .868 | .496 |
| Query Imminence (15s) | XGB+Seq-Trans | .843 | .524 |
| Query Type | Seq-GRU | .771 | .696 |

All LLMs perform near chance (AUC .42–.60), demonstrating that behavioral prediction requires task-specific training on telemetry data.

## Ethics and Privacy

- All student data is de-identified with randomized IDs
- No personally identifiable information is included
- Query text content is not included in the released features
- The study was approved under IRB protocol #[REDACTED]

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@inproceedings{anonymous2026tutortrace,
  title={TutorTrace: A Behavioral Telemetry Dataset and Benchmark for AI-Assisted Programming Education},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems: Datasets and Benchmarks Track},
  year={2026}
}
```
