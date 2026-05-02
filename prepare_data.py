"""
TutorTrace Data Preparation
============================
Reads the manifest, checks for existing derived files,
and generates any that are missing from raw telemetry.

Usage:
    python prepare_data.py                  # generate missing files
    python prepare_data.py --force          # regenerate everything
    python prepare_data.py --only segments  # only generate segments

Pipeline:
    raw_telemetry.json
        → behavioral_sequences/segments.csv      (auto-segmenter)
        → observable_metrics/windows.csv          (sliding window features)
        → observable_metrics/queries.csv          (per-query features)

Tasks:
    1. Next behavioral state (window-level, 5-class)
    2. Error imminence (window-level, binary at multiple horizons)
    3. Query imminence (window-level, binary at multiple horizons)
    4. Post-query improvement (query-level, binary)
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'manifest.yaml')

from behavioral_classifier.auto_segmenter import auto_segment_events


# ── Event types ───────────────────────────────────────────────

CODE_TYPES     = {'CODE_TYPE', 'CODE_DELETE', 'CODE_DELETE_SELECTION',
                  'CODE_PASTE', 'CODE_COPY', 'CODE_UNDO', 'CODE_REDO',
                  'CODE_SELECT', 'CODE_INDENT', 'CODE_CUT', 'CODE_UNKNOWN'}
TERMINAL_TYPES = {'TERMINAL_RUN', 'TERMINAL_OUTPUT'}
ERROR_TYPES    = {'TERMINAL_ERROR'}
QUERY_TYPES    = {'CHAT_QUERY', 'CHAT_SEND'}
RESPONSE_TYPES = {'CHAT_RESPONSE', 'CHAT_RECEIVE'}
NOISE_TYPES    = {'MOUSE_MOVE', 'WINDOW_RESIZE'}

WINDOW_SIZE_S = 30
WINDOW_STEP_S = 5
MIN_IDLE_S    = 3


# ══════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"  ERROR: {MANIFEST_PATH} not found")
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_raw(path):
    """Load raw telemetry JSON into a tidy DataFrame."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    rows = []
    for sid, sdata in raw.items():
        for ev in sdata.get('events', []):
            payload = ev.get('payload', {}) or {}
            text_len = 0
            if isinstance(payload, dict):
                text = payload.get('text') or payload.get('content') or payload.get('message') or ''
                text_len = len(text) if text else (payload.get('length', 0) or 0)

            rows.append({
                'student_id':        str(sid),
                'timestamp_ms':      ev.get('timestamp'),
                'type':              ev.get('type'),
                'test_passed_count': payload.get('passed_count') if isinstance(payload, dict) else None,
                'test_total_count':  payload.get('total_tests') if isinstance(payload, dict) else None,
                'payload_text_len':  text_len,
                'payload_visible':   payload.get('visible') if isinstance(payload, dict) else None,
            })

    df = pd.DataFrame(rows)
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    df = df.dropna(subset=['timestamp_ms'])
    df['timestamp_s'] = df['timestamp_ms'] / 1000.0
    return df.sort_values(['student_id', 'timestamp_ms']).reset_index(drop=True)


def derived_paths(name):
    return {
        'segments': os.path.join(DATASET_DIR, 'behavioral_sequences', f'{name}_segments.csv'),
        'windows':  os.path.join(DATASET_DIR, 'observable_metrics', 'window_level', f'{name}_windows.csv'),
        'queries':  os.path.join(DATASET_DIR, 'observable_metrics', 'query_level', f'{name}_queries.csv'),
    }


# ══════════════════════════════════════════════════════════════
#  STEP 1: SEGMENTS
# ══════════════════════════════════════════════════════════════

def generate_segments(df, out_path):
    print("    [1/3] Generating behavioral segments...")
    rows = []
    students = df['student_id'].unique()

    for i, sid in enumerate(students):
        s = df[df['student_id'] == sid]
        if len(s) < 5:
            continue

        start_ms = s['timestamp_ms'].min()
        end_ms = s['timestamp_ms'].max()
        dur_ms = end_ms - start_ms
        if dur_ms < 10000:
            continue

        events_list = []
        for _, r in s.iterrows():
            payload = {}
            if pd.notna(r.get('test_passed_count')):
                payload['passed_count'] = int(r['test_passed_count'])
            if pd.notna(r.get('test_total_count')):
                payload['total_tests'] = int(r['test_total_count'])
            events_list.append({
                'timestamp': r['timestamp_ms'],
                'type': r['type'],
                'payload': payload if payload else {},
            })

        try:
            segs = auto_segment_events(events_list, start_ms, dur_ms)
            for j, seg in enumerate(segs or []):
                behavior = seg.get('suggestedBehavior') or {}
                rows.append({
                    'student_id':       sid,
                    'segment_index':    j,
                    'behavioral_state': behavior.get('id', 'unknown'),
                    'thinking_subtype': seg.get('suggestedThinkingSubcategory', ''),
                    'start_time_ms':    seg.get('startTime', 0),
                    'end_time_ms':      seg.get('endTime', 0),
                    'duration_s':       round((seg.get('endTime', 0) - seg.get('startTime', 0)) / 1000, 2),
                })
        except Exception as e:
            print(f"      WARN: {sid} failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(students)} students...")

    seg_df = pd.DataFrame(rows)
    save(seg_df, out_path)

    n = seg_df['student_id'].nunique()
    print(f"      {n} students → {len(seg_df):,} segments")
    dist = seg_df['behavioral_state'].value_counts()
    for state, count in dist.items():
        print(f"        {state}: {count} ({count/len(seg_df)*100:.1f}%)")

    return seg_df


# ══════════════════════════════════════════════════════════════
#  STEP 2: WINDOW-LEVEL FEATURES
# ══════════════════════════════════════════════════════════════

def generate_windows(df, seg_df, out_path):
    print("    [2/3] Generating window-level features...")
    rows = []
    students = df['student_id'].unique()

    for i, sid in enumerate(students):
        s = df[df['student_id'] == sid]
        if len(s) < 5:
            continue

        t0 = s['timestamp_s'].min()
        t1 = s['timestamp_s'].max()
        if t1 - t0 < WINDOW_SIZE_S * 2:
            continue

        student_segs = seg_df[seg_df['student_id'] == sid] if seg_df is not None else pd.DataFrame()

        # Pre-compute event times for this student
        q_times = s[s['type'].isin(QUERY_TYPES)]['timestamp_s'].tolist()
        resp_times_list = s[s['type'].isin(RESPONSE_TYPES)]['timestamp_s'].tolist()
        err_times = s[s['type'].isin(ERROR_TYPES)]['timestamp_s'].tolist()

        ws = t0
        while ws + WINDOW_SIZE_S <= t1:
            we = ws + WINDOW_SIZE_S
            win = s[(s['timestamp_s'] >= ws) & (s['timestamp_s'] < we)]
            cum = s[s['timestamp_s'] < we]

            # Skip chat-typing windows
            if len(win[win['type'].isin({'CHAT_TYPE', 'CHAT_PASTE'})]) > 0:
                ws += WINDOW_STEP_S
                continue

            f = {
                'student_id': sid,
                'window_start_s': round(ws - t0, 2),
                'window_end_s': round(we - t0, 2),
            }

            # ── Layer 1: Raw counts ──
            code  = win[win['type'].isin(CODE_TYPES)]
            runs  = win[win['type'].isin(TERMINAL_TYPES)]
            errs  = win[win['type'].isin(ERROR_TYPES)]
            tests = win[win['type'] == 'TEST_CASE_RESULT']
            queries = win[win['type'].isin(QUERY_TYPES)]

            f['code_events'] = len(code)
            f['terminal_runs'] = len(runs)
            f['terminal_errors'] = len(errs)
            f['test_results'] = len(tests)
            f['query_count'] = len(queries)

            active = win[~win['type'].isin(NOISE_TYPES)]
            if len(active) >= 2:
                ts = sorted(active['timestamp_s'].tolist())
                gaps = np.diff(ts)
                f['event_density'] = round(len(active) / WINDOW_SIZE_S, 4)
                f['longest_idle_s'] = round(float(gaps.max()), 2)
                idle_gaps = gaps[gaps >= MIN_IDLE_S]
                f['thinking_time_s'] = round(float(idle_gaps.sum()), 2) if len(idle_gaps) > 0 else 0
            else:
                f['event_density'] = 0
                f['longest_idle_s'] = WINDOW_SIZE_S
                f['thinking_time_s'] = WINDOW_SIZE_S

            # ── Layer 2: Observable metrics ──
            dur = max(1, we - t0)
            cum_code = cum[cum['type'].isin(CODE_TYPES)]
            cum_q = cum[cum['type'].isin(QUERY_TYPES)]

            f['cum_code_rate'] = round(len(cum_code) / dur, 4)
            f['cum_query_rate'] = round(len(cum_q) / dur, 4)
            f['query_count_so_far'] = len(cum_q)
            f['time_since_session_start_s'] = round(we - t0, 2)

            inserts = len(win[win['type'].isin({'CODE_TYPE', 'CODE_PASTE'})])
            deletes = len(win[win['type'].isin({'CODE_DELETE', 'CODE_DELETE_SELECTION'})])
            f['net_code_growth'] = inserts - deletes
            f['delete_ratio'] = round(deletes / max(1, inserts + deletes), 4)

            prior_q_cum = cum[cum['type'].isin(QUERY_TYPES)]
            last_q_t = prior_q_cum['timestamp_s'].max() if len(prior_q_cum) > 0 else 0
            f['time_since_last_query_s'] = round(we - last_q_t, 2) if last_q_t > 0 else round(we - t0, 2)

            error_times_win = errs['timestamp_s'].tolist()
            code_times_win = code['timestamp_s'].tolist()
            f['error_self_fix'] = sum(1 for et in error_times_win if any(0 < ct - et <= 30 for ct in code_times_win))

            # Prior no-effort rate
            prior_q = [qt for qt in q_times if qt < we]
            if len(prior_q) >= 2:
                no_effort_count = 0
                for qi_idx in range(len(prior_q) - 1):
                    q1 = prior_q[qi_idx]
                    resp_after = [rt for rt in resp_times_list if rt > q1]
                    if resp_after:
                        r1 = min(resp_after)
                        q2 = prior_q[qi_idx + 1]
                        btwn = s[(s['timestamp_s'] > r1) & (s['timestamp_s'] < q2)]
                        if len(btwn[btwn['type'].isin(CODE_TYPES)]) == 0 and len(btwn[btwn['type'].isin(TERMINAL_TYPES)]) == 0:
                            no_effort_count += 1
                f['prior_no_effort_rate'] = round(no_effort_count / (len(prior_q) - 1), 4)
            else:
                f['prior_no_effort_rate'] = 0

            # ── Layer 3: Behavioral sequence features ──
            if len(student_segs) > 0:
                win_start_ms = (ws - t0) * 1000
                win_end_ms = (we - t0) * 1000

                active_segs = student_segs[
                    (student_segs['start_time_ms'] < win_end_ms) &
                    (student_segs['end_time_ms'] > win_start_ms)
                ]

                if len(active_segs) > 0:
                    last_seg = active_segs.iloc[-1]
                    f['current_state'] = last_seg['behavioral_state']
                    f['prev_state'] = active_segs.iloc[-2]['behavioral_state'] if len(active_segs) >= 2 else ''
                    f['segments_in_window'] = len(active_segs)

                    for state in ['thinking', 'implementing', 'debugging', 'seekingHelp', 'testing']:
                        n_state = len(active_segs[active_segs['behavioral_state'] == state])
                        f[f'pct_{state}'] = round(n_state / len(active_segs), 4)
                else:
                    f['current_state'] = ''
                    f['prev_state'] = ''
                    f['segments_in_window'] = 0
                    for state in ['thinking', 'implementing', 'debugging', 'seekingHelp', 'testing']:
                        f[f'pct_{state}'] = 0

            # ── Labels ──

            # Task 1: Next behavioral state
            if len(student_segs) > 0:
                win_end_ms = (we - t0) * 1000
                future_segs = student_segs[student_segs['start_time_ms'] >= win_end_ms]
                if len(future_segs) > 0:
                    next_seg = future_segs.iloc[0]
                    f['label_next_state'] = next_seg['behavioral_state']

                    if next_seg['behavioral_state'] == 'thinking' and next_seg.get('thinking_subtype'):
                        f['label_next_thinking_subtype'] = next_seg['thinking_subtype']
                    else:
                        f['label_next_thinking_subtype'] = None
                else:
                    f['label_next_state'] = None
                    f['label_next_thinking_subtype'] = None

                for k in [3, 5]:
                    if len(future_segs) >= k:
                        f[f'label_next_{k}_states'] = ','.join(future_segs.iloc[:k]['behavioral_state'].tolist())
                    else:
                        f[f'label_next_{k}_states'] = None
            else:
                f['label_next_state'] = None
                f['label_next_thinking_subtype'] = None
                for k in [3, 5]:
                    f[f'label_next_{k}_states'] = None

            # Task 2: Error imminence
            for horizon in [15, 30, 60]:
                f[f'label_error_imminence_{horizon}s'] = int(any(we <= et <= we + horizon for et in err_times))

            # Task 3: Query imminence
            for horizon in [5, 10, 15, 30, 45, 60]:
                f[f'label_query_imminence_{horizon}s'] = int(any(we <= qt <= we + horizon for qt in q_times))

            # Time to next query (regression)
            future_q = [qt for qt in q_times if qt >= we]
            if future_q:
                f['label_time_to_next_query_s'] = round(min(future_q) - we, 2)
            else:
                f['label_time_to_next_query_s'] = None

            rows.append(f)
            ws += WINDOW_STEP_S

        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(students)} students...")

    win_df = pd.DataFrame(rows)
    save(win_df, out_path)
    print(f"      {win_df['student_id'].nunique()} students → {len(win_df):,} windows, {len(win_df.columns)} columns")
    return win_df


# ══════════════════════════════════════════════════════════════
#  STEP 3: QUERY-LEVEL FEATURES
# ══════════════════════════════════════════════════════════════

def _compute_window_metrics(w, dur):
    """Compute observable metrics for a window of events."""
    f = {}

    code = w[w['type'].isin(CODE_TYPES)]
    code_types = w[w['type'] == 'CODE_TYPE']
    code_deletes = w[w['type'].isin({'CODE_DELETE', 'CODE_DELETE_SELECTION'})]
    code_pastes = w[w['type'] == 'CODE_PASTE']
    code_copies = w[w['type'] == 'CODE_COPY']
    code_undos = w[w['type'] == 'CODE_UNDO']

    f['code_edits'] = len(code)
    f['code_edit_rate'] = round(len(code) / max(0.1, dur), 4)
    f['chars_inserted'] = len(code_types)
    f['chars_deleted'] = len(code_deletes)
    f['code_deletes'] = len(code_deletes)
    f['net_code_growth'] = len(code_types) + len(code_pastes) - len(code_deletes)
    f['delete_type_ratio'] = round(len(code_deletes) / max(1, len(code)), 4)
    f['code_pastes'] = len(code_pastes)
    f['code_copies'] = len(code_copies)
    f['code_undos'] = len(code_undos)

    runs = w[w['type'].isin(TERMINAL_TYPES)]
    errs = w[w['type'].isin(ERROR_TYPES)]
    f['terminal_runs'] = len(runs)
    f['terminal_errors'] = len(errs)

    max_consec = 0
    consec = 0
    for _, e in w.iterrows():
        if e['type'] in ERROR_TYPES:
            consec += 1
            max_consec = max(max_consec, consec)
        elif e['type'] in TERMINAL_TYPES:
            consec = 0
    f['max_consecutive_errors'] = max_consec

    run_times = runs['timestamp_s'].tolist()
    if len(run_times) >= 2:
        gaps = np.diff(run_times)
        f['mean_time_between_runs_s'] = round(float(np.mean(gaps)), 2)
    else:
        f['mean_time_between_runs_s'] = 0

    error_self_fix = 0
    error_ai_fix = 0
    error_to_edit_times = []
    error_reading_times = []

    for _, err in errs.iterrows():
        after = w[w['timestamp_s'] > err['timestamp_s']]
        next_edit = after[after['type'].isin(CODE_TYPES)]
        next_query = after[after['type'].isin(QUERY_TYPES)]
        next_active = after[after['type'].isin(CODE_TYPES | TERMINAL_TYPES | QUERY_TYPES)]

        if len(next_active) > 0:
            error_reading_times.append(next_active.iloc[0]['timestamp_s'] - err['timestamp_s'])
        if len(next_edit) > 0:
            error_to_edit_times.append(next_edit.iloc[0]['timestamp_s'] - err['timestamp_s'])

        if len(next_edit) > 0 and (len(next_query) == 0 or next_edit.iloc[0]['timestamp_s'] < next_query.iloc[0]['timestamp_s']):
            error_self_fix += 1
        elif len(next_query) > 0:
            error_ai_fix += 1

    f['error_self_fix'] = error_self_fix
    f['error_ai_fix'] = error_ai_fix
    f['error_reading_time_s'] = round(float(np.mean(error_reading_times)), 2) if error_reading_times else 0
    f['error_to_edit_s'] = round(float(np.mean(error_to_edit_times)), 2) if error_to_edit_times else 0

    failed_tests = w[(w['type'] == 'TEST_CASE_RESULT')]
    failed_test_count = 0
    failed_test_self_fix = 0
    failed_test_ai_fix = 0
    failed_test_to_edit_times = []

    for _, ft in failed_tests.iterrows():
        if pd.notna(ft.get('test_passed_count')) and pd.notna(ft.get('test_total_count')):
            if ft['test_passed_count'] < ft['test_total_count']:
                failed_test_count += 1
                after = w[w['timestamp_s'] > ft['timestamp_s']]
                next_edit = after[after['type'].isin(CODE_TYPES)]
                next_query = after[after['type'].isin(QUERY_TYPES)]

                if len(next_edit) > 0:
                    failed_test_to_edit_times.append(next_edit.iloc[0]['timestamp_s'] - ft['timestamp_s'])
                if len(next_edit) > 0 and (len(next_query) == 0 or next_edit.iloc[0]['timestamp_s'] < next_query.iloc[0]['timestamp_s']):
                    failed_test_self_fix += 1
                elif len(next_query) > 0:
                    failed_test_ai_fix += 1

    f['failed_test_count'] = failed_test_count
    f['failed_test_self_fix'] = failed_test_self_fix
    f['failed_test_ai_fix'] = failed_test_ai_fix
    f['failed_test_to_edit_s'] = round(float(np.mean(failed_test_to_edit_times)), 2) if failed_test_to_edit_times else 0

    active = w[~w['type'].isin(NOISE_TYPES)]
    if len(active) >= 2:
        ts = sorted(active['timestamp_s'].tolist())
        gaps = np.diff(ts)
        f['longest_idle_s'] = round(float(gaps.max()), 2)
        idle_gaps = gaps[gaps >= MIN_IDLE_S]
        f['thinking_time_s'] = round(float(idle_gaps.sum()), 2) if len(idle_gaps) > 0 else 0
    else:
        f['longest_idle_s'] = round(dur, 2)
        f['thinking_time_s'] = round(dur, 2)

    tab_events = w[w['type'] == 'TAB_STATE']
    f['tab_switches'] = len(tab_events)
    tab_hidden_time = 0
    tab_list = tab_events.index.tolist()
    for ti in range(len(tab_list)):
        row = w.loc[tab_list[ti]]
        if row.get('payload_visible') == False and ti < len(tab_list) - 1:
            next_tab = w.loc[tab_list[ti + 1]]
            tab_hidden_time += next_tab['timestamp_s'] - row['timestamp_s']
    f['tab_hidden_time_s'] = round(tab_hidden_time, 2)

    f['duration_s'] = round(dur, 2)

    return f


def _compute_segment_features(student_segs, win_start_ms, win_end_ms, prefix=''):
    """Compute behavioral time features from segments overlapping a window."""
    f = {}
    state_times = {'implementing': 0, 'debugging': 0, 'testing': 0, 'seekingHelp': 0, 'thinking': 0}
    subtype_times = {'thinking-task': 0, 'thinking-llm': 0, 'thinking-error': 0, 'thinking-code': 0}

    if len(student_segs) == 0:
        for st in state_times:
            f[f'{prefix}{st}_time_s'] = 0
        for st in subtype_times:
            f[f'{prefix}{st.replace("-", "_")}_s'] = 0
        return f

    for _, seg in student_segs.iterrows():
        o_start = max(seg['start_time_ms'], win_start_ms)
        o_end = min(seg['end_time_ms'], win_end_ms)
        if o_end > o_start:
            o_s = (o_end - o_start) / 1000
            state = seg['behavioral_state']
            if state in state_times:
                state_times[state] += o_s
            subtype = seg.get('thinking_subtype', '')
            if subtype and subtype in subtype_times:
                subtype_times[subtype] += o_s

    for st, t in state_times.items():
        f[f'{prefix}{st}_time_s'] = round(t, 2)
    for st, t in subtype_times.items():
        f[f'{prefix}{st.replace("-", "_")}_s'] = round(t, 2)

    return f


def generate_queries(df, seg_df, out_path):
    print("    [3/3] Generating query-level features...")
    rows = []

    for sid in df['student_id'].unique():
        s = df[df['student_id'] == sid].sort_values('timestamp_s')
        if len(s) < 5:
            continue

        t0 = s['timestamp_s'].min()
        t1 = s['timestamp_s'].max()

        queries = s[s['type'].isin(QUERY_TYPES)]
        if len(queries) == 0:
            continue

        responses = s[s['type'].isin(RESPONSE_TYPES)]
        tests = s[s['type'] == 'TEST_CASE_RESULT']
        q_indices = queries.index.tolist()

        last_test = tests.iloc[-1] if len(tests) > 0 else None
        final_passed = int(last_test['test_passed_count']) if last_test is not None and pd.notna(last_test.get('test_passed_count')) else 0
        final_total = int(last_test['test_total_count']) if last_test is not None and pd.notna(last_test.get('test_total_count')) else 0
        task_completed = 1 if final_total > 0 and final_passed == final_total else 0

        student_segs = seg_df[seg_df['student_id'] == sid] if seg_df is not None else pd.DataFrame()

        for qi, row_idx in enumerate(q_indices):
            q_ts = s.loc[row_idx]['timestamp_s']
            is_first = qi == 0
            is_last = qi == len(q_indices) - 1

            prior_resp = responses[responses['timestamp_s'] < q_ts]
            win_start = prior_resp['timestamp_s'].max() if len(prior_resp) > 0 else t0
            has_prior_ai = len(prior_resp) > 0
            prev_q_ts = s.loc[q_indices[qi - 1]]['timestamp_s'] if qi > 0 else None

            w = s[(s['timestamp_s'] >= win_start) & (s['timestamp_s'] < q_ts)]
            dur = max(0.1, q_ts - win_start)

            pre = _compute_window_metrics(w, dur)

            pre_seg_start_ms = (win_start - t0) * 1000
            pre_seg_end_ms = (q_ts - t0) * 1000
            pre_behav = _compute_segment_features(student_segs, pre_seg_start_ms, pre_seg_end_ms, prefix='')

            response_reading_time = 0
            chat_to_code_latency = 0
            if has_prior_ai:
                last_resp_ts = prior_resp['timestamp_s'].max()
                post_resp = w[w['timestamp_s'] > last_resp_ts]
                post_resp_active = post_resp[post_resp['type'].isin(CODE_TYPES | TERMINAL_TYPES)]
                if len(post_resp_active) > 0:
                    response_reading_time = round(post_resp_active.iloc[0]['timestamp_s'] - last_resp_ts, 2)
                else:
                    response_reading_time = round(q_ts - last_resp_ts, 2)
                post_resp_code = post_resp[post_resp['type'].isin(CODE_TYPES)]
                if len(post_resp_code) > 0:
                    chat_to_code_latency = round(post_resp_code.iloc[0]['timestamp_s'] - last_resp_ts, 2)

            query_text_len = s.loc[row_idx].get('payload_text_len', 0) or 0

            post_responses = responses[responses['timestamp_s'] > q_ts]
            ai_response_len = 0
            if len(post_responses) > 0:
                ai_resp_row = post_responses.iloc[0]
                ai_response_len = ai_resp_row.get('payload_text_len', 0) or 0

            tests_before = tests[tests['timestamp_s'] <= q_ts]
            lt = tests_before.iloc[-1] if len(tests_before) > 0 else None
            test_passed = int(lt['test_passed_count']) if lt is not None and pd.notna(lt.get('test_passed_count')) else 0
            test_total = int(lt['test_total_count']) if lt is not None and pd.notna(lt.get('test_total_count')) else 0

            rows.append({
                'student_id': sid,
                'query_index': qi + 1,
                'total_queries': len(q_indices),
                'is_first_query': 1 if is_first else 0,
                'time_since_session_start_s': round(q_ts - t0, 2),
                'time_since_last_query_s': round(q_ts - prev_q_ts, 2) if prev_q_ts else 0,
                'pre_duration_s': round(dur, 2),
                'pre_code_edits': pre['code_edits'],
                'pre_code_edit_rate': pre['code_edit_rate'],
                'pre_chars_inserted': pre['chars_inserted'],
                'pre_chars_deleted': pre['chars_deleted'],
                'pre_code_deletes': pre['code_deletes'],
                'pre_net_code_growth': pre['net_code_growth'],
                'pre_delete_type_ratio': pre['delete_type_ratio'],
                'pre_terminal_runs': pre['terminal_runs'],
                'pre_terminal_errors': pre['terminal_errors'],
                'pre_max_consecutive_errors': pre['max_consecutive_errors'],
                'pre_mean_time_between_runs_s': pre['mean_time_between_runs_s'],
                'pre_error_self_fix': pre['error_self_fix'],
                'pre_error_ai_fix': pre['error_ai_fix'],
                'pre_error_reading_time_s': pre['error_reading_time_s'],
                'pre_error_to_edit_s': pre['error_to_edit_s'],
                'pre_failed_test_count': pre['failed_test_count'],
                'pre_failed_test_self_fix': pre['failed_test_self_fix'],
                'pre_failed_test_ai_fix': pre['failed_test_ai_fix'],
                'pre_failed_test_to_edit_s': pre['failed_test_to_edit_s'],
                'pre_longest_idle_s': pre['longest_idle_s'],
                'thinking_time_s': pre['thinking_time_s'],
                'pre_tab_switches': pre['tab_switches'],
                'pre_tab_hidden_time_s': pre['tab_hidden_time_s'],
                'pre_response_reading_time_s': response_reading_time,
                'pre_chat_to_code_latency_s': chat_to_code_latency,
                'pre_time_in_editor_s': pre_behav.get('implementing_time_s', 0),
                'pre_time_in_terminal_s': pre_behav.get('testing_time_s', 0) + pre_behav.get('debugging_time_s', 0),
                'pre_time_in_chat_s': pre_behav.get('seekingHelp_time_s', 0),
                'pre_time_in_task_s': pre_behav.get('thinking_task_s', 0),
                'pre_time_in_tests_s': pre_behav.get('testing_time_s', 0),
                'implementing_time_s': pre_behav.get('implementing_time_s', 0),
                'debugging_time_s': pre_behav.get('debugging_time_s', 0),
                'testing_time_s': pre_behav.get('testing_time_s', 0),
                'seeking_help_time_s': pre_behav.get('seekingHelp_time_s', 0),
                'thinking_task_s': pre_behav.get('thinking_task_s', 0),
                'thinking_llm_s': pre_behav.get('thinking_llm_s', 0),
                'thinking_error_s': pre_behav.get('thinking_error_s', 0),
                'thinking_code_s': pre_behav.get('thinking_code_s', 0),
                'query_length_chars': query_text_len,
                'ai_response_length_chars': ai_response_len,
                'test_passed_at_query': test_passed,
                'test_total_at_query': test_total,
                'task_completed': task_completed,
                'final_test_passed': final_passed,
                'final_test_total': final_total,
                'session_duration_s': round(t1 - t0, 2),
            })

    q_df = pd.DataFrame(rows)

    # ── Task 4 Label: Post-query improvement ──
    # Did test_passed increase from previous query to this query?
    q_df['prev_test_passed'] = q_df.groupby('student_id')['test_passed_at_query'].shift(1)
    q_df['label_post_query_improvement'] = None
    mask = q_df['prev_test_passed'].notna()
    q_df.loc[mask, 'label_post_query_improvement'] = (
        q_df.loc[mask, 'test_passed_at_query'] > q_df.loc[mask, 'prev_test_passed']
    ).astype(int)
    q_df = q_df.drop(columns=['prev_test_passed'])

    save(q_df, out_path)
    print(f"      {q_df['student_id'].nunique()} students → {len(q_df)} queries, {len(q_df.columns)} columns")

    if 'label_post_query_improvement' in q_df.columns:
        valid = q_df['label_post_query_improvement'].dropna()
        if len(valid) > 0:
            print(f"      Post-query improvement: {len(valid)} valid, {int(valid.sum())} positive ({valid.mean()*100:.1f}%)")

    return q_df


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def check(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True, f"✓ ({os.path.getsize(path):,} bytes)"
    return False, "✗ missing"


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def process_deployment(name, config, force=False, only=None):
    print(f"\n  ┌─ {name} [{config.get('split', '')}]")

    raw_path = os.path.join(DATASET_DIR, config['raw_telemetry'])
    paths = derived_paths(name)

    if not os.path.exists(raw_path):
        print(f"  │  ✗ Raw telemetry not found: {raw_path}")
        print(f"  └─ SKIPPED")
        return

    seg_ok, seg_s = check(paths['segments'])
    win_ok, win_s = check(paths['windows'])
    qry_ok, qry_s = check(paths['queries'])

    print(f"  │  Segments: {seg_s}")
    print(f"  │  Windows:  {win_s}")
    print(f"  │  Queries:  {qry_s}")

    need_seg = force or not seg_ok
    need_win = force or not win_ok
    need_qry = force or not qry_ok

    if only == 'segments':
        need_win, need_qry = False, False
    elif only == 'windows':
        need_seg, need_qry = False, False
    elif only == 'queries':
        need_seg, need_win = False, False

    if not need_seg and not need_win and not need_qry:
        print(f"  └─ Up to date")
        return

    print(f"  │")
    print(f"  │  Loading raw telemetry...")
    df = load_raw(raw_path)
    n_stu = df['student_id'].nunique()
    n_evt = len(df)
    n_ai = df[df['type'].isin(QUERY_TYPES)]['student_id'].nunique()
    print(f"      {n_stu} students, {n_evt:,} events, {n_ai} AI users")

    seg_df = None
    if need_seg:
        print(f"  │")
        seg_df = generate_segments(df, paths['segments'])
    elif seg_ok and (need_win or need_qry):
        print(f"  │  Loading existing segments...")
        seg_df = pd.read_csv(paths['segments'])
        seg_df['student_id'] = seg_df['student_id'].astype(str)

    if need_win:
        print(f"  │")
        generate_windows(df, seg_df, paths['windows'])

    if need_qry:
        print(f"  │")
        generate_queries(df, seg_df, paths['queries'])

    print(f"  └─ Done")


def main():
    parser = argparse.ArgumentParser(description='TutorTrace Data Preparation')
    parser.add_argument('--force', action='store_true', help='Regenerate all files')
    parser.add_argument('--only', choices=['segments', 'windows', 'queries'], help='Only generate one type')
    args = parser.parse_args()

    print("=" * 60)
    print("  TUTORTRACE DATA PREPARATION")
    print("=" * 60)

    manifest = load_manifest()
    deployments = manifest.get('deployments', {})

    if not deployments:
        print("  No deployments in manifest.")
        return

    print(f"  {len(deployments)} deployment(s)")

    for name, config in deployments.items():
        if not config.get('enabled', True):
            print(f"\n  ── {name}: disabled, skipping")
            continue
        process_deployment(name, config, force=args.force, only=args.only)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print("=" * 60)
    for name, config in deployments.items():
        if not config.get('enabled', True):
            continue
        p = derived_paths(name)
        s_ok, _ = check(p['segments'])
        w_ok, _ = check(p['windows'])
        q_ok, _ = check(p['queries'])
        icon = '✓' if all([s_ok, w_ok, q_ok]) else '⚠'
        s_i = '✓' if s_ok else '✗'
        w_i = '✓' if w_ok else '✗'
        q_i = '✓' if q_ok else '✗'
        print(f"  {icon} {name} [{config.get('split','')}]: seg {s_i}  win {w_i}  qry {q_i}")

    print()


if __name__ == '__main__':
    main()