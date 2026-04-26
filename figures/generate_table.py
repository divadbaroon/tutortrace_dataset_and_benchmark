"""Generate HTML tables from benchmark results.json and dataset stats."""
import json
import sys
import os


def generate_html(results_path, output_path, stats_path=None):
    with open(results_path) as f:
        results = json.load(f)

    stats = None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TutorTrace Benchmark Results</title>
<style>
  body { font-family: 'Times New Roman', serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #111; }
  h1 { font-size: 22px; text-align: center; margin-bottom: 6px; }
  .subtitle { font-size: 13px; text-align: center; color: #666; margin-bottom: 30px; }
  h2 { font-size: 16px; margin-top: 40px; border-bottom: 2px solid #111; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0 30px 0; font-size: 13px; }
  th, td { padding: 6px 12px; text-align: center; }
  th { border-top: 2px solid #111; border-bottom: 1px solid #111; font-weight: bold; }
  td { border-bottom: none; }
  tr:last-child td { border-bottom: 2px solid #111; }
  .task-header td { border-top: 1px solid #999; }
  .task-label { text-align: left; font-weight: 600; }
  .model-name { text-align: left; padding-left: 20px; }
  .best { font-weight: bold; }
  caption { font-size: 13px; text-align: left; margin-bottom: 8px; font-style: italic; }
  .overview { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0 30px 0; }
  .overview-card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
  .overview-card h3 { font-size: 13px; margin: 0 0 12px 0; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-row { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; }
  .stat-label { color: #555; }
  .stat-value { font-weight: 600; font-family: 'Courier New', monospace; }
  .stat-big { font-size: 28px; font-weight: 800; color: #111; margin-bottom: 4px; }
  .stat-desc { font-size: 11px; color: #888; }
  .highlight-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }
  .highlight-card { text-align: center; padding: 16px; border: 1px solid #eee; border-radius: 8px; }
</style>
</head>
<body>
"""

    # ── Dataset Overview ──
    if stats:
        d1 = stats.get('Deployment 1 (train)', {})
        d2 = stats.get('Deployment 2 (test)', {})
        c = stats.get('combined', {})

        html += '<h2>Dataset Overview</h2>\n'

        # Table 1: Deployment comparison
        html += '<table>\n'
        html += '<caption>Deployment summary.</caption>\n'
        html += '<tr><th style="text-align:left">Metric</th><th>Deployment 1 (Train)</th><th>Deployment 2 (Test)</th><th>Combined</th></tr>\n'

        dep_rows = [
            ('Students', d1.get('students',0), d2.get('students',0), c.get('students',0)),
            ('Students using AI', d1.get('students_with_ai',0), d2.get('students_with_ai',0), c.get('students_with_ai',0)),
            ('Tasks completed', f"{d1.get('tasks_completed',0)}/{d1.get('students',0)} ({d1.get('tasks_completed',0)/max(1,d1.get('students',0))*100:.0f}%)", f"{d2.get('tasks_completed',0)}/{d2.get('students',0)} ({d2.get('tasks_completed',0)/max(1,d2.get('students',0))*100:.0f}%)", f"{c.get('tasks_completed',0)}/{c.get('students',0)}"),
            ('Avg session (min)', d1.get('avg_session_min',0), d2.get('avg_session_min',0), '—'),
            ('Median session (min)', d1.get('median_session_min',0), d2.get('median_session_min',0), '—'),
            ('Total minutes', f"{d1.get('total_minutes',0):.0f}", f"{d2.get('total_minutes',0):.0f}", f"{c.get('total_minutes',0):.0f}"),
        ]
        for label, v1, v2, v3 in dep_rows:
            html += f'<tr><td style="text-align:left">{label}</td><td>{v1}</td><td>{v2}</td><td>{v3}</td></tr>\n'
        html += '</table>\n'

        # Table 2: Telemetry breakdown
        html += '<table>\n'
        html += '<caption>Telemetry event counts.</caption>\n'
        html += '<tr><th style="text-align:left">Event Category</th><th>Deployment 1</th><th>Deployment 2</th><th>Combined</th></tr>\n'

        d1_types = d1.get('event_type_counts', {})
        d2_types = d2.get('event_type_counts', {})

        telem_rows = [
            ('Raw events (all)', d1.get('total_events',0), d2.get('total_events',0), c.get('total_events',0)),
            ('Raw events (excl. mouse/clicks)', d1.get('events_no_mouse',0), d2.get('events_no_mouse',0), c.get('events_no_mouse',0)),
            ('Code edits (CODE_TYPE)', d1_types.get('CODE_TYPE',0), d2_types.get('CODE_TYPE',0), d1_types.get('CODE_TYPE',0)+d2_types.get('CODE_TYPE',0)),
            ('Code deletes', d1_types.get('CODE_DELETE',0)+d1_types.get('CODE_DELETE_SELECTION',0), d2_types.get('CODE_DELETE',0)+d2_types.get('CODE_DELETE_SELECTION',0), d1_types.get('CODE_DELETE',0)+d1_types.get('CODE_DELETE_SELECTION',0)+d2_types.get('CODE_DELETE',0)+d2_types.get('CODE_DELETE_SELECTION',0)),
            ('Terminal runs', d1_types.get('TERMINAL_RUN',0), d2_types.get('TERMINAL_RUN',0), d1_types.get('TERMINAL_RUN',0)+d2_types.get('TERMINAL_RUN',0)),
            ('Terminal output', d1_types.get('TERMINAL_OUTPUT',0), d2_types.get('TERMINAL_OUTPUT',0), d1_types.get('TERMINAL_OUTPUT',0)+d2_types.get('TERMINAL_OUTPUT',0)),
            ('Terminal errors', d1.get('total_errors',0), d2.get('total_errors',0), c.get('total_errors',0)),
            ('Test results', d1.get('total_test_results',0), d2.get('total_test_results',0), c.get('total_test_results',0)),
            ('AI queries', d1.get('total_queries',0), d2.get('total_queries',0), c.get('total_queries',0)),
            ('AI responses', d1.get('total_responses',0), d2.get('total_responses',0), c.get('total_responses',0)),
            ('Chat typing events', d1_types.get('CHAT_TYPE',0), d2_types.get('CHAT_TYPE',0), d1_types.get('CHAT_TYPE',0)+d2_types.get('CHAT_TYPE',0)),
            ('Tab state changes', d1_types.get('TAB_STATE',0), d2_types.get('TAB_STATE',0), d1_types.get('TAB_STATE',0)+d2_types.get('TAB_STATE',0)),
            ('Mouse moves', d1_types.get('MOUSE_MOVE',0), d2_types.get('MOUSE_MOVE',0), d1_types.get('MOUSE_MOVE',0)+d2_types.get('MOUSE_MOVE',0)),
        ]
        for label, v1, v2, v3 in telem_rows:
            html += f'<tr><td style="text-align:left">{label}</td><td>{v1:,}</td><td>{v2:,}</td><td>{v3:,}</td></tr>\n'
        html += '</table>\n'

        # Table 3: Behavioral segments
        seg_dist_d1 = d1.get('segment_distribution', {})
        seg_dist_d2 = d2.get('segment_distribution', {})
        total_d1 = sum(seg_dist_d1.values()) if seg_dist_d1 else 0
        total_d2 = sum(seg_dist_d2.values()) if seg_dist_d2 else 0

        if total_d1 > 0 or total_d2 > 0:
            html += '<table>\n'
            html += '<caption>Behavioral segment distribution (auto-segmenter output).</caption>\n'
            html += '<tr><th style="text-align:left">Behavioral State</th><th>Deployment 1</th><th>%</th><th>Deployment 2</th><th>%</th></tr>\n'

            for state in ['thinking', 'debugging', 'implementing', 'seekingHelp', 'testing']:
                c1 = seg_dist_d1.get(state, 0)
                c2 = seg_dist_d2.get(state, 0)
                p1 = f"{c1/max(1,total_d1)*100:.1f}%" if total_d1 else '—'
                p2 = f"{c2/max(1,total_d2)*100:.1f}%" if total_d2 else '—'
                html += f'<tr><td style="text-align:left">{state}</td><td>{c1:,}</td><td>{p1}</td><td>{c2:,}</td><td>{p2}</td></tr>\n'

            html += f'<tr style="font-weight:600"><td style="text-align:left">Total</td><td>{total_d1:,}</td><td></td><td>{total_d2:,}</td><td></td></tr>\n'
            html += '</table>\n'

        # Table 4: Thinking subtypes
        subtypes = d1.get('thinking_subtypes', {})
        if subtypes:
            html += '<table>\n'
            html += '<caption>Thinking subtype distribution (Deployment 1).</caption>\n'
            html += '<tr><th style="text-align:left">Subtype</th><th>Count</th><th>%</th></tr>\n'

            total_thinking = sum(subtypes.values())
            for st in ['thinking-error', 'thinking-code', 'thinking-llm', 'thinking-task']:
                count = subtypes.get(st, 0)
                pct = f"{count/max(1,total_thinking)*100:.1f}%"
                html += f'<tr><td style="text-align:left; font-family:monospace; font-size:12px">{st}</td><td>{count:,}</td><td>{pct}</td></tr>\n'

            html += f'<tr style="font-weight:600"><td style="text-align:left">Total</td><td>{total_thinking:,}</td><td></td></tr>\n'
            html += '</table>\n'

    # ── Table 1: Main Results ──
    tasks = [
        ('next_behavioral_state', 'Next behavioral state (5-class)'),
        ('thinking_subtype', 'Thinking subtype (conditional)'),
        ('query_imminence_60s', 'Query imminence (60s)'),
        ('query_with_no_effort', 'Query with no effort'),
    ]
    models = ['Majority', 'LogReg', 'RandomForest', 'XGBoost', 'LSTM']
    layers = ['Raw telemetry', '+Observable', '+Behav. sequences']

    html += '<h2>Main Results</h2>\n'
    html += '<table>\n'
    html += '<caption>Held-out AUC / Macro F1 across all tasks and three feature abstraction layers.</caption>\n'
    html += '<tr><th style="text-align:left">Task</th><th style="text-align:left">Model</th>'
    for l in layers:
        html += f'<th colspan="2">{l}</th>'
    html += '</tr>\n'
    html += '<tr><th></th><th></th>'
    for l in layers:
        html += '<th>AUC</th><th>F1</th>'
    html += '</tr>\n'

    for task_key, task_name in tasks:
        if task_key not in results:
            continue
        task_data = results[task_key]

        best_auc = 0
        for m in models:
            if m in task_data:
                for l in layers:
                    if l in task_data[m]:
                        auc = task_data[m][l].get('auc', 0) or 0
                        best_auc = max(best_auc, auc)

        for mi, model in enumerate(models):
            if model not in task_data:
                continue
            html += '<tr'
            if mi == 0:
                html += ' class="task-header"'
            html += '>'

            if mi == 0:
                html += f'<td class="task-label" rowspan="{len(models)}">{task_name}</td>'
            html += f'<td class="model-name">{model}</td>'

            for layer in layers:
                if layer in task_data[model]:
                    auc = task_data[model][layer].get('auc', 0) or 0
                    f1 = task_data[model][layer].get('macro_f1', 0) or 0
                    bold = ' class="best"' if abs(auc - best_auc) < 0.001 and auc > 0.5 else ''
                    html += f'<td{bold}>{auc:.3f}</td><td>{f1:.3f}</td>'
                else:
                    html += '<td>—</td><td>—</td>'
            html += '</tr>\n'

    html += '</table>\n'

    # ── Table 2: Compact ──
    html += '<h2>Compact Summary (Best Model per Task)</h2>\n'
    html += '<table>\n'
    html += '<tr><th style="text-align:left">Task</th><th>Raw AUC/F1</th><th>+Observable AUC/F1</th><th>+Behav. Seq. AUC/F1</th><th>ΔAUC</th></tr>\n'

    for task_key, task_name in tasks:
        if task_key not in results:
            continue
        best = {}
        for layer in layers:
            best_auc = 0
            best_f1 = 0
            for m in models:
                if m in results[task_key] and layer in results[task_key][m]:
                    auc = results[task_key][m][layer].get('auc', 0) or 0
                    f1 = results[task_key][m][layer].get('macro_f1', 0) or 0
                    if auc > best_auc:
                        best_auc = auc
                        best_f1 = f1
            best[layer] = (best_auc, best_f1)

        raw_auc = best[layers[0]][0]
        seq_auc = best[layers[2]][0]
        delta = seq_auc - raw_auc

        html += f'<tr><td style="text-align:left">{task_name}</td>'
        for layer in layers:
            a, f = best[layer]
            html += f'<td>{a:.3f} / {f:.3f}</td>'
        html += f'<td>+{delta:.3f}</td></tr>\n'

    html += '</table>\n'

    # ── Table 3: Query Imminence Horizons ──
    html += '<h2>Query Imminence Across Horizons</h2>\n'
    html += '<table>\n'
    html += '<tr><th>Horizon</th><th>Best Model</th><th>Raw</th><th>+Observable</th><th>+Behav. Seq.</th></tr>\n'

    for h in [15, 30, 45, 60]:
        key = f'query_imminence_{h}s'
        if key not in results:
            continue
        best_per_layer = {}
        best_model = ''
        for layer in layers:
            best_auc = 0
            for m in models:
                if m in results[key] and layer in results[key][m]:
                    auc = results[key][m][layer].get('auc', 0) or 0
                    if auc > best_auc:
                        best_auc = auc
                        if layer == layers[2]:
                            best_model = m
            best_per_layer[layer] = best_auc

        html += f'<tr><td>{h}s</td><td>{best_model}</td>'
        for layer in layers:
            html += f'<td>{best_per_layer[layer]:.3f}</td>'
        html += '</tr>\n'

    html += '</table>\n'
    html += '</body></html>'

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Generated: {output_path}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    results_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root_dir, 'benchmark', 'results.json')
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, 'results_tables.html')
    stats_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(script_dir, 'dataset_stats.json')
    generate_html(results_path, output_path, stats_path)