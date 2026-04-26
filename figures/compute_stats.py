"""Compute dataset statistics from raw telemetry JSON files."""
import json
import sys
import os
import numpy as np


def compute_stats(json_path, segments_path=None):
    """Compute comprehensive stats for one deployment."""
    with open(json_path, 'r') as f:
        raw = json.load(f)

    stats = {
        'students': len(raw),
        'students_with_ai': 0,
        'total_events': 0,
        'events_no_mouse': 0,
        'total_minutes': 0,
        'total_queries': 0,
        'total_responses': 0,
        'total_errors': 0,
        'total_terminal_runs': 0,
        'total_code_edits': 0,
        'total_code_deletes': 0,
        'total_test_results': 0,
        'total_segments': 0,
        'tasks_completed': 0,
        'event_type_counts': {},
    }

    MOUSE_CLICK_TYPES = {'MOUSE_MOVE', 'MOUSE_CLICK', 'WINDOW_RESIZE', 'PANEL_RESIZE'}
    CODE_TYPES = {'CODE_TYPE', 'CODE_DELETE', 'CODE_DELETE_SELECTION',
                  'CODE_PASTE', 'CODE_COPY', 'CODE_UNDO', 'CODE_REDO',
                  'CODE_SELECT', 'CODE_INDENT', 'CODE_CUT', 'CODE_UNKNOWN'}

    session_durations = []

    for sid, sdata in raw.items():
        events = sdata.get('events', [])
        if len(events) < 2:
            continue

        stats['total_events'] += len(events)

        has_query = False
        task_completed = False
        timestamps = []

        for e in events:
            etype = e.get('type', '')
            payload = e.get('payload', {}) or {}
            ts = e.get('timestamp', 0)
            timestamps.append(ts)

            # Count by type
            stats['event_type_counts'][etype] = stats['event_type_counts'].get(etype, 0) + 1

            # Exclude mouse/click
            if etype not in MOUSE_CLICK_TYPES:
                stats['events_no_mouse'] += 1

            # Specific counts
            if etype in ('CHAT_QUERY', 'CHAT_SEND'):
                stats['total_queries'] += 1
                has_query = True
            elif etype in ('CHAT_RESPONSE', 'CHAT_RECEIVE'):
                stats['total_responses'] += 1
            elif etype == 'TERMINAL_ERROR':
                stats['total_errors'] += 1
            elif etype in ('TERMINAL_RUN', 'TERMINAL_OUTPUT'):
                stats['total_terminal_runs'] += 1
            elif etype in CODE_TYPES:
                stats['total_code_edits'] += 1
                if etype in ('CODE_DELETE', 'CODE_DELETE_SELECTION'):
                    stats['total_code_deletes'] += 1
            elif etype == 'TEST_CASE_RESULT':
                stats['total_test_results'] += 1
                passed = payload.get('passed_count', 0) or 0
                total = payload.get('total_tests', 0) or 0
                if total > 0 and passed == total:
                    task_completed = True

        if has_query:
            stats['students_with_ai'] += 1
        if task_completed:
            stats['tasks_completed'] += 1

        if timestamps:
            dur_min = (max(timestamps) - min(timestamps)) / 1000 / 60
            session_durations.append(dur_min)
            stats['total_minutes'] += dur_min

    stats['total_minutes'] = round(stats['total_minutes'], 1)
    stats['avg_session_min'] = round(np.mean(session_durations), 1) if session_durations else 0
    stats['median_session_min'] = round(np.median(session_durations), 1) if session_durations else 0

    # Load segments if available
    if segments_path and os.path.exists(segments_path):
        import pandas as pd
        segs = pd.read_csv(segments_path)
        stats['total_segments'] = len(segs)
        stats['segment_distribution'] = segs['behavioral_state'].value_counts().to_dict()
        if 'thinking_subtype' in segs.columns:
            thinking = segs[segs['behavioral_state'] == 'thinking']
            stats['thinking_subtypes'] = thinking['thinking_subtype'].value_counts().to_dict()

    return stats


def print_stats(name, stats):
    """Print stats for one deployment."""
    print(f"\n  {name}")
    print(f"  {'─' * 50}")
    print(f"  Students:           {stats['students']}")
    print(f"  Students using AI:  {stats['students_with_ai']}")
    print(f"  Tasks completed:    {stats['tasks_completed']}/{stats['students']} ({stats['tasks_completed']/max(1,stats['students'])*100:.0f}%)")
    print(f"  Total minutes:      {stats['total_minutes']} ({stats['avg_session_min']} avg, {stats['median_session_min']} median)")
    print(f"")
    print(f"  Raw events (all):       {stats['total_events']:,}")
    print(f"  Raw events (no mouse):  {stats['events_no_mouse']:,}")
    print(f"  Code edits:             {stats['total_code_edits']:,}")
    print(f"  Terminal runs:          {stats['total_terminal_runs']:,}")
    print(f"  Terminal errors:        {stats['total_errors']:,}")
    print(f"  Test results:           {stats['total_test_results']:,}")
    print(f"  AI queries:             {stats['total_queries']:,}")
    print(f"  AI responses:           {stats['total_responses']:,}")
    print(f"  Behavioral segments:    {stats['total_segments']:,}")

    if 'segment_distribution' in stats:
        print(f"\n  Segment distribution:")
        for state, count in sorted(stats['segment_distribution'].items(), key=lambda x: -x[1]):
            pct = count / max(1, stats['total_segments']) * 100
            print(f"    {state:<20s} {count:>5,} ({pct:.1f}%)")

    if 'thinking_subtypes' in stats:
        print(f"\n  Thinking subtypes:")
        for subtype, count in sorted(stats['thinking_subtypes'].items(), key=lambda x: -x[1]):
            print(f"    {subtype:<20s} {count:>5,}")

    print(f"\n  Event types:")
    for etype, count in sorted(stats['event_type_counts'].items(), key=lambda x: -x[1]):
        print(f"    {etype:<30s} {count:>8,}")


def generate_stats_json(deployments, output_path):
    """Compute stats for all deployments and save as JSON."""
    all_stats = {}
    for name, paths in deployments.items():
        stats = compute_stats(paths['json'], paths.get('segments'))
        all_stats[name] = stats
        print_stats(name, stats)

    # Combined
    combined = {
        'students': sum(s['students'] for s in all_stats.values()),
        'students_with_ai': sum(s['students_with_ai'] for s in all_stats.values()),
        'total_events': sum(s['total_events'] for s in all_stats.values()),
        'events_no_mouse': sum(s['events_no_mouse'] for s in all_stats.values()),
        'total_minutes': round(sum(s['total_minutes'] for s in all_stats.values()), 1),
        'total_queries': sum(s['total_queries'] for s in all_stats.values()),
        'total_responses': sum(s['total_responses'] for s in all_stats.values()),
        'total_errors': sum(s['total_errors'] for s in all_stats.values()),
        'total_terminal_runs': sum(s['total_terminal_runs'] for s in all_stats.values()),
        'total_code_edits': sum(s['total_code_edits'] for s in all_stats.values()),
        'total_test_results': sum(s['total_test_results'] for s in all_stats.values()),
        'total_segments': sum(s['total_segments'] for s in all_stats.values()),
        'tasks_completed': sum(s['tasks_completed'] for s in all_stats.values()),
    }

    print(f"\n  {'═' * 50}")
    print(f"  COMBINED")
    print(f"  {'═' * 50}")
    print(f"  Students:           {combined['students']}")
    print(f"  Students using AI:  {combined['students_with_ai']}")
    print(f"  Total events:       {combined['total_events']:,}")
    print(f"  Events (no mouse):  {combined['events_no_mouse']:,}")
    print(f"  Total minutes:      {combined['total_minutes']}")
    print(f"  Code edits:         {combined['total_code_edits']:,}")
    print(f"  AI queries:         {combined['total_queries']:,}")
    print(f"  Terminal errors:    {combined['total_errors']:,}")
    print(f"  Segments:           {combined['total_segments']:,}")

    all_stats['combined'] = combined

    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return all_stats


if __name__ == '__main__':
    # Default paths for the repo structure
    # Root is one level up from figures/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(root, 'dataset')

    deployments = {
        'Deployment 1 (train)': {
            'json': os.path.join(dataset_dir, 'raw_telemetry', 'deployment_1.json'),
            'segments': os.path.join(dataset_dir, 'behavioral_sequences', 'deployment_1_segments.csv'),
        },
        'Deployment 2 (test)': {
            'json': os.path.join(dataset_dir, 'raw_telemetry', 'deployment_2.json'),
            'segments': os.path.join(dataset_dir, 'behavioral_sequences', 'deployment_2_segments.csv'),
        },
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_stats.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate_stats_json(deployments, output_path)