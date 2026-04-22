"""
TutorTrace Auto-Segmenter
=========================
Converts raw IDE telemetry events into behavioral segments.

Each segment has a behavioral state:
  - thinking    (reading task, reviewing code, pausing)
  - implementing (writing/editing code)
  - debugging   (editing code after an unresolved error)
  - seekingHelp (typing/sending a query to the AI tutor)
  - testing     (running code, viewing terminal output, test results)

Pipeline steps:
  1. buildMajorSegments        — group by event category, split code on 6s gaps
  2. fillGapsWithThinking      — gaps ≥3s → Thinking; smaller → extend adjacent
  3. mergeShortTestingSegments — testing <1.5s → absorb into neighbor
  4. absorbPreQueryPauses      — short thinking before chat → metadata
  5. applyErrorState           — implementing + unresolved error → debugging
  6. applyThinkingSubtypes     — task / llm / error / code
  7. postProcessSegments       — fix nulls + merge consecutive same-behavior

Usage:
    from auto_segmenter import auto_segment_events
    segments = auto_segment_events(events, start_time_ms, duration_ms)
"""

import copy

# ══════════════════════════════════════════════════════════════
#  BEHAVIORAL CODES
# ══════════════════════════════════════════════════════════════

BEHAVIORAL_CODES = {
    'thinking':     {'id': 'thinking',     'label': 'Thinking'},
    'implementing': {'id': 'implementing', 'label': 'Implementing'},
    'debugging':    {'id': 'debugging',    'label': 'Debugging'},
    'seekingHelp':  {'id': 'seekingHelp',  'label': 'Seeking Help'},
    'testing':      {'id': 'testing',      'label': 'Testing'},
    'unknown':      {'id': 'unknown',      'label': 'Unknown'},
}


def get_behavior(behavior_id):
    """Look up a behavioral code by ID."""
    return BEHAVIORAL_CODES.get(behavior_id, BEHAVIORAL_CODES['unknown'])


# ══════════════════════════════════════════════════════════════
#  EVENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════

CODE_EVENTS = {
    'CODE_TYPE', 'CODE_DELETE', 'CODE_DELETE_SELECTION',
    'CODE_PASTE', 'CODE_CUT', 'CODE_UNDO', 'CODE_REDO', 'CODE_INDENT',
    'CODE_UNKNOWN',
}

TERMINAL_EVENTS = {
    'TERMINAL_RUN', 'TEST_CASE_RESULT', 'TERMINAL_ERROR', 'TERMINAL_OUTPUT',
}

CHAT_INPUT_EVENTS = {
    'CHAT_TYPE', 'CHAT_PASTE', 'CHAT_DELETE', 'CHAT_QUERY',
}

CHAT_RESPONSE_EVENTS = {
    'CHAT_RESPONSE',
}


def get_event_category(event):
    """Classify an event into a behavioral category."""
    event_type = event.get('type', '')
    if event_type in CODE_EVENTS:
        return 'code'
    if event_type in TERMINAL_EVENTS:
        return 'terminal'
    if event_type in CHAT_INPUT_EVENTS:
        return 'chatInput'
    if event_type in CHAT_RESPONSE_EVENTS:
        return 'chatResponse'
    return None


def get_behavior_for_category(category):
    """Map event category to behavioral state."""
    mapping = {
        'code':         'implementing',
        'terminal':     'testing',
        'chatInput':    'seekingHelp',
        'chatResponse': 'thinking',
    }
    return get_behavior(mapping.get(category, 'unknown'))


# ══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def auto_segment_events(events, start_time, duration):
    """
    Auto-segment events into behavioral chunks.

    Args:
        events: list of event dicts with 'timestamp' and 'type'
        start_time: session start time in milliseconds
        duration: session duration in milliseconds

    Returns:
        list of segment dicts with behavioral state assignments
    """
    if not events or len(events) == 0:
        return []

    # Normalize timestamps relative to start
    normalized = []
    for e in events:
        ne = dict(e)
        ne['time'] = e['timestamp'] - start_time
        normalized.append(ne)

    # Run the 7-step pipeline
    major_segments = build_major_segments(normalized, duration)
    all_segments = fill_gaps_with_thinking(major_segments, duration)
    merged_testing = merge_short_testing_segments(all_segments)
    merged_segments = absorb_pre_query_pauses(merged_testing)
    with_error_state = apply_error_state(merged_segments, normalized)
    with_subtypes = apply_thinking_subtypes(with_error_state, normalized)
    return post_process_segments(with_subtypes, normalized)


# ══════════════════════════════════════════════════════════════
#  STEP 1: BUILD MAJOR SEGMENTS
# ══════════════════════════════════════════════════════════════

def build_major_segments(events, duration):
    """Group events by category, split code segments on 6s gaps."""
    INACTIVITY_SPLIT_MS = 6000
    segments = []

    current_segment = None
    current_category = None
    last_event_time = None
    seg_counter = 0

    for event in events:
        category = get_event_category(event)
        if category is None:
            continue

        # Skip chat responses if we're not in a chat input segment
        if category == 'chatResponse' and current_category and current_category != 'chatInput':
            continue

        # Split code segments on inactivity gaps
        if (category == current_category and category == 'code'
                and last_event_time is not None):
            gap = event['time'] - last_event_time
            if gap >= INACTIVITY_SPLIT_MS:
                current_segment['endTime'] = last_event_time
                segments.append(current_segment)
                current_segment = {
                    'id': f'segment-{len(segments)}-{event["time"]}',
                    'startTime': event['time'],
                    'endTime': None,
                    'behavior': None,
                    'suggestedBehavior': get_behavior_for_category(category),
                    'suggestedThinkingSubcategory': None,
                    'confidence': None,
                    'preQueryPause': None,
                }

        # Category change
        if category != current_category:
            if current_segment is not None:
                if current_category == 'chatInput' and category == 'chatResponse':
                    current_segment['endTime'] = event['time']
                else:
                    current_segment['endTime'] = last_event_time
                segments.append(current_segment)

            current_category = category
            current_segment = {
                'id': f'segment-{len(segments)}-{event["time"]}',
                'startTime': event['time'],
                'endTime': None,
                'behavior': None,
                'suggestedBehavior': get_behavior_for_category(category),
                'suggestedThinkingSubcategory': 'thinking-llm' if category == 'chatResponse' else None,
                'confidence': None,
                'preQueryPause': None,
            }

        last_event_time = event['time']

    if current_segment is not None:
        current_segment['endTime'] = last_event_time
        segments.append(current_segment)

    return segments


# ══════════════════════════════════════════════════════════════
#  STEP 2: FILL GAPS WITH THINKING
# ══════════════════════════════════════════════════════════════

def fill_gaps_with_thinking(major_segments, duration):
    """Insert Thinking segments in gaps ≥3s; extend adjacent for smaller gaps."""
    MIN_THINKING_MS = 3000

    if len(major_segments) == 0:
        return [{
            'id': 'thinking-0-0',
            'startTime': 0,
            'endTime': duration,
            'behavior': None,
            'suggestedBehavior': get_behavior('thinking'),
            'suggestedThinkingSubcategory': None,
            'confidence': None,
            'preQueryPause': None,
        }]

    all_segments = []
    seg_index = 0

    # Handle gap before first segment
    first = major_segments[0]
    if first['startTime'] >= MIN_THINKING_MS:
        all_segments.append({
            'id': 'thinking-start-0',
            'startTime': 0,
            'endTime': first['startTime'],
            'behavior': None,
            'suggestedBehavior': get_behavior('thinking'),
            'suggestedThinkingSubcategory': None,
            'confidence': None,
            'preQueryPause': None,
        })
    elif first['startTime'] > 0:
        first['startTime'] = 0

    # Process segments and gaps between them
    for i in range(len(major_segments)):
        current = major_segments[i]
        all_segments.append(current)

        if i < len(major_segments) - 1:
            next_seg = major_segments[i + 1]
            gap_duration = next_seg['startTime'] - current['endTime']

            if gap_duration >= MIN_THINKING_MS:
                if current['startTime'] == current['endTime']:
                    current['endTime'] = next_seg['startTime']
                else:
                    all_segments.append({
                        'id': f'thinking-gap-{seg_index}',
                        'startTime': current['endTime'],
                        'endTime': next_seg['startTime'],
                        'behavior': None,
                        'suggestedBehavior': get_behavior('thinking'),
                        'suggestedThinkingSubcategory': None,
                        'confidence': None,
                        'preQueryPause': None,
                    })
                    seg_index += 1
            elif gap_duration > 0:
                current['endTime'] = next_seg['startTime']

    # Handle gap after last segment
    last = all_segments[-1]
    end_gap = duration - last['endTime']

    if end_gap >= MIN_THINKING_MS:
        if last['startTime'] == last['endTime']:
            last['endTime'] = duration
        else:
            all_segments.append({
                'id': f'thinking-end-{duration}',
                'startTime': last['endTime'],
                'endTime': duration,
                'behavior': None,
                'suggestedBehavior': get_behavior('thinking'),
                'suggestedThinkingSubcategory': None,
                'confidence': None,
                'preQueryPause': None,
            })
    elif end_gap > 0:
        last['endTime'] = duration

    return all_segments


# ══════════════════════════════════════════════════════════════
#  STEP 3: MERGE SHORT TESTING SEGMENTS
# ══════════════════════════════════════════════════════════════

def merge_short_testing_segments(segments):
    """Absorb testing segments shorter than 1.5s into neighbors."""
    MAX_MICRO_TESTING_MS = 1500

    if len(segments) < 2:
        return segments

    result = []

    for i in range(len(segments)):
        current = segments[i]
        next_seg = segments[i + 1] if i < len(segments) - 1 else None

        behavior_id = (current.get('suggestedBehavior') or {}).get('id')
        is_testing = behavior_id == 'testing' or behavior_id is None
        seg_duration = current['endTime'] - current['startTime']
        is_micro = seg_duration < MAX_MICRO_TESTING_MS

        if is_testing and is_micro and next_seg:
            next_seg['startTime'] = current['startTime']
        elif is_testing and is_micro and not next_seg and len(result) > 0:
            result[-1]['endTime'] = current['endTime']
        else:
            result.append(current)

    return result


# ══════════════════════════════════════════════════════════════
#  STEP 4: ABSORB PRE-QUERY PAUSES
# ══════════════════════════════════════════════════════════════

def absorb_pre_query_pauses(segments):
    """Short thinking before chat input → absorbed as metadata."""
    MAX_PRE_QUERY_PAUSE_MS = 5000

    if len(segments) < 3:
        return segments

    result = []

    for i in range(len(segments)):
        prev = result[-1] if len(result) > 0 else None
        current = segments[i]
        next_seg = segments[i + 1] if i < len(segments) - 1 else None

        is_thinking = (current.get('suggestedBehavior') or {}).get('id') == 'thinking'
        pause_duration = current['endTime'] - current['startTime']
        is_short = pause_duration <= MAX_PRE_QUERY_PAUSE_MS

        prev_is_code_or_debug = prev and (prev.get('suggestedBehavior') or {}).get('id') in (
            'implementing', 'debugging'
        )
        next_is_chat = next_seg and (next_seg.get('suggestedBehavior') or {}).get('id') == 'seekingHelp'

        if is_thinking and is_short and prev_is_code_or_debug and next_is_chat:
            next_seg['preQueryPause'] = pause_duration
            prev['endTime'] = current['endTime']
        else:
            result.append(current)

    return result


# ══════════════════════════════════════════════════════════════
#  STEP 5: APPLY ERROR STATE
# ══════════════════════════════════════════════════════════════

def apply_error_state(segments, events):
    """Reclassify implementing → debugging when there's an unresolved error."""
    has_unresolved_error = False

    result = []
    for segment in segments:
        seg = dict(segment)

        # Find events in this segment
        seg_events = [e for e in events
                      if e['time'] >= seg['startTime'] and e['time'] < seg['endTime']]

        has_error = any(e['type'] == 'TERMINAL_ERROR' for e in seg_events)
        has_failed_test = any(
            e['type'] == 'TEST_CASE_RESULT'
            and e.get('payload', {}).get('total_tests', 0) > 0
            and e.get('payload', {}).get('passed_count', 0) < e.get('payload', {}).get('total_tests', 0)
            for e in seg_events
        )
        has_passed_all = any(
            e['type'] == 'TEST_CASE_RESULT'
            and e.get('payload', {}).get('passed_count', 0) == e.get('payload', {}).get('total_tests', 0)
            and e.get('payload', {}).get('total_tests', 0) > 0
            for e in seg_events
        )

        if has_error or has_failed_test:
            has_unresolved_error = True
        if has_passed_all:
            has_unresolved_error = False

        behavior_id = (seg.get('suggestedBehavior') or {}).get('id')
        if behavior_id == 'implementing' and has_unresolved_error:
            seg['suggestedBehavior'] = get_behavior('debugging')

        result.append(seg)

    return result


# ══════════════════════════════════════════════════════════════
#  STEP 6: APPLY THINKING SUBTYPES
# ══════════════════════════════════════════════════════════════

def apply_thinking_subtypes(segments, events):
    """
    Classify each Thinking segment with a subtype:
      thinking-task  — no code/run/chat activity has occurred yet
      thinking-llm   — previous segment was seekingHelp
      thinking-error — most recent terminal run had an unresolved error
      thinking-code  — residual: reviewing own code
    """
    has_any_activity = False

    # Pre-compute failed and passed run times
    failed_run_times = set()
    passed_run_times = set()

    for e in events:
        if e['type'] == 'TERMINAL_ERROR':
            failed_run_times.add(e['time'])
        if e['type'] == 'TEST_CASE_RESULT':
            payload = e.get('payload', {})
            passed = payload.get('passed_count', 0)
            total = payload.get('total_tests', 0)
            if total > 0 and passed < total:
                failed_run_times.add(e['time'])
            if total > 0 and passed == total:
                passed_run_times.add(e['time'])

    result = []
    for i, segment in enumerate(segments):
        seg = dict(segment)
        behavior_id = (seg.get('suggestedBehavior') or {}).get('id')

        if behavior_id != 'thinking':
            if behavior_id in ('implementing', 'debugging', 'seekingHelp'):
                has_any_activity = True
            result.append(seg)
            continue

        # Rule 1: thinking-task (no activity yet)
        if not has_any_activity:
            has_any_activity = True
            seg['suggestedThinkingSubcategory'] = 'thinking-task'
            result.append(seg)
            continue

        # Rule 2: thinking-llm (previous segment was seekingHelp)
        prev_seg = segments[i - 1] if i > 0 else None
        if prev_seg and (prev_seg.get('suggestedBehavior') or {}).get('id') == 'seekingHelp':
            seg['suggestedThinkingSubcategory'] = 'thinking-llm'
            result.append(seg)
            continue

        # Rule 3: thinking-error (unresolved error before this segment)
        recent_failed = [t for t in failed_run_times if t < seg['startTime']]
        recent_passed = [t for t in passed_run_times if t < seg['startTime']]

        if recent_failed:
            latest_failed = max(recent_failed)
            latest_passed = max(recent_passed) if recent_passed else None

            if latest_passed is None or latest_failed > latest_passed:
                seg['suggestedThinkingSubcategory'] = 'thinking-error'
                result.append(seg)
                continue

        # Rule 4: thinking-code (residual)
        seg['suggestedThinkingSubcategory'] = 'thinking-code'
        result.append(seg)

    return result


# ══════════════════════════════════════════════════════════════
#  STEP 7: POST-PROCESS SEGMENTS
# ══════════════════════════════════════════════════════════════

def post_process_segments(segments, events):
    """
    1. Fix null-behavior segments (terminal → testing, else → thinking)
    2. Merge consecutive same-behavior segments
    3. Re-index segment IDs
    """
    thinking_behavior = get_behavior('thinking')
    testing_behavior = get_behavior('testing')

    # Step 1: fix null behaviors
    working = []
    for seg in segments:
        s = dict(seg)
        behavior_id = None
        if s.get('suggestedBehavior'):
            behavior_id = s['suggestedBehavior'].get('id')
        if not behavior_id and s.get('behavior'):
            behavior_id = s['behavior'].get('id')

        if behavior_id is None:
            has_terminal = any(
                e['time'] >= s['startTime'] and e['time'] < s['endTime']
                and e['type'] in TERMINAL_EVENTS
                for e in events
            )
            s['suggestedBehavior'] = testing_behavior if has_terminal else thinking_behavior
            s['isSystemEvent'] = True

        working.append(s)

    # Step 2: merge consecutive same-behavior
    if len(working) < 2:
        return working

    merged = [dict(working[0])]
    for i in range(1, len(working)):
        prev = merged[-1]
        curr = working[i]

        prev_id = (prev.get('suggestedBehavior') or {}).get('id') or (prev.get('behavior') or {}).get('id')
        curr_id = (curr.get('suggestedBehavior') or {}).get('id') or (curr.get('behavior') or {}).get('id')

        if prev_id == curr_id:
            prev['endTime'] = curr['endTime']
            if curr.get('preQueryPause'):
                prev['preQueryPause'] = curr['preQueryPause']
        else:
            merged.append(dict(curr))

    # Step 3: re-index
    for i, seg in enumerate(merged):
        seg['id'] = f'segment-final-{i}-{seg["startTime"]}'

    return merged


# ══════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def compute_observables(events, start_time):
    """
    Compute observable metrics from a list of events.
    Returns dict with 'before' key containing metric values.
    
    This is a simplified version — the full VizPI implementation
    computes more detailed metrics in the JavaScript frontend.
    """
    if not events:
        return {'before': {}}

    metrics = {}
    code_events = [e for e in events if e.get('type', '') in CODE_EVENTS]
    terminal_events = [e for e in events if e.get('type', '') in TERMINAL_EVENTS]
    error_events = [e for e in events if e.get('type', '') == 'TERMINAL_ERROR']
    query_events = [e for e in events if e.get('type', '') in CHAT_INPUT_EVENTS]

    duration_ms = max(1, (events[-1]['timestamp'] - events[0]['timestamp'])) if len(events) > 1 else 1

    metrics['codeEdit_count'] = len(code_events)
    metrics['codeEdit_per_min'] = round(len(code_events) / (duration_ms / 60000), 2) if duration_ms > 0 else 0
    metrics['terminalRun_count'] = len(terminal_events)
    metrics['terminalError_count'] = len(error_events)
    metrics['chatQuery_count'] = len(query_events)
    metrics['period_duration_ms'] = duration_ms

    # Time since last code edit
    if code_events:
        last_code_time = max(e['timestamp'] for e in code_events)
        metrics['time_since_code_edit_ms'] = events[-1]['timestamp'] - last_code_time
    else:
        metrics['time_since_code_edit_ms'] = 999999

    # Mouse moves per minute
    mouse_events = [e for e in events if e.get('type', '') == 'MOUSE_MOVE']
    metrics['mouseMove_per_min'] = round(len(mouse_events) / (duration_ms / 60000), 2) if duration_ms > 0 else 0

    return {'before': metrics}


def observables_to_features(obs, prefix='obs_'):
    """Convert observable metrics dict to flat feature dict with prefix."""
    features = {}
    for key, value in obs.items():
        features[f'{prefix}{key}'] = value
    return features