"""
Window Size Ablation for TutorTrace Benchmark.

Tests multiple window sizes (15s, 30s, 45s, 60s) with proportional step sizes.
For each size: regenerates windows, runs benchmark, saves results.

Usage:
  cd tutortrace_dataset_and_benchmark
  python3 benchmark/run_window_ablation.py
"""

import os
import re
import subprocess
import json
import shutil


WINDOW_CONFIGS = [
    {'window_s': 15, 'step_s': 3},
    {'window_s': 30, 'step_s': 5},
    {'window_s': 45, 'step_s': 8},
    {'window_s': 60, 'step_s': 10},
]


def find_and_replace_window_size(filepath, window_s, step_s):
    """Replace WINDOW_SIZE_S and WINDOW_STEP_S in prepare_data.py."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Save original values for restoration
    orig_window = re.search(r'WINDOW_SIZE_S\s*=\s*(\d+)', content)
    orig_step = re.search(r'WINDOW_STEP_S\s*=\s*(\d+)', content)

    if not orig_window or not orig_step:
        print(f"  ERROR: Could not find WINDOW_SIZE_S or WINDOW_STEP_S in {filepath}")
        return None, None

    orig_window_val = int(orig_window.group(1))
    orig_step_val = int(orig_step.group(1))

    content = re.sub(r'WINDOW_SIZE_S\s*=\s*\d+', f'WINDOW_SIZE_S = {window_s}', content)
    content = re.sub(r'WINDOW_STEP_S\s*=\s*\d+', f'WINDOW_STEP_S = {step_s}', content)

    with open(filepath, 'w') as f:
        f.write(content)

    return orig_window_val, orig_step_val


def restore_window_size(filepath, window_s, step_s):
    """Restore original WINDOW_SIZE_S and WINDOW_STEP_S."""
    with open(filepath, 'r') as f:
        content = f.read()

    content = re.sub(r'WINDOW_SIZE_S\s*=\s*\d+', f'WINDOW_SIZE_S = {window_s}', content)
    content = re.sub(r'WINDOW_STEP_S\s*=\s*\d+', f'WINDOW_STEP_S = {step_s}', content)

    with open(filepath, 'w') as f:
        f.write(content)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prepare_data_path = os.path.join(root, 'prepare_data.py')
    results_dir = os.path.join(root, 'benchmark', 'window_ablation_results')
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(prepare_data_path):
        print(f"  ERROR: {prepare_data_path} not found")
        return

    print("=" * 60)
    print("  WINDOW SIZE ABLATION")
    print("=" * 60)
    print(f"  Configs: {WINDOW_CONFIGS}")
    print()

    all_results = {}

    for config in WINDOW_CONFIGS:
        window_s = config['window_s']
        step_s = config['step_s']

        print(f"\n{'=' * 60}")
        print(f"  WINDOW: {window_s}s (step: {step_s}s)")
        print(f"{'=' * 60}\n")

        # Modify prepare_data.py
        orig_window, orig_step = find_and_replace_window_size(
            prepare_data_path, window_s, step_s
        )
        if orig_window is None:
            continue

        try:
            # Regenerate windows with --force
            print(f"  Regenerating windows ({window_s}s / {step_s}s)...")
            result = subprocess.run(
                ['python3', prepare_data_path, '--force'],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  ERROR in prepare_data.py: {result.stderr[:500]}")
                continue

            # Run benchmark
            print(f"  Running benchmark...")
            result = subprocess.run(
                ['python3', os.path.join(root, 'benchmark', 'run_benchmark.py')],
                cwd=root,
                capture_output=True,
                text=True,
            )

            # Save output
            output_path = os.path.join(results_dir, f'window_{window_s}s_output.txt')
            with open(output_path, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write('\n\nSTDERR:\n')
                    f.write(result.stderr)

            # Copy results.json
            src_results = os.path.join(root, 'benchmark', 'results.json')
            if os.path.exists(src_results):
                dst_results = os.path.join(results_dir, f'window_{window_s}s_results.json')
                with open(src_results) as f:
                    data = json.load(f)
                with open(dst_results, 'w') as f:
                    json.dump(data, f, indent=2)
                all_results[f'{window_s}s'] = data

            print(f"  ✓ {window_s}s complete")

            if result.returncode != 0:
                print(f"  WARNING: benchmark returned code {result.returncode}")

        finally:
            # Always restore original values
            restore_window_size(prepare_data_path, orig_window, orig_step)

    # Regenerate windows at original size
    print(f"\n  Restoring original windows ({orig_window}s / {orig_step}s)...")
    subprocess.run(
        ['python3', prepare_data_path, '--force'],
        cwd=root,
        capture_output=True,
        text=True,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  WINDOW ABLATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {results_dir}/")
    for f in sorted(os.listdir(results_dir)):
        print(f"    {f}")


if __name__ == '__main__':
    main()