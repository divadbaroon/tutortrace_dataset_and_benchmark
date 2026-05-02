"""
TutorTrace Full Benchmark Suite
================================
Runs all benchmark configurations automatically:
  1. Setup A: D1 train → D2 test (main results)
  2. Per-deployment: D1 train → each deployment individually
  3. Setup B: D1+D3+D4 train → D2+D5 test (multi-instructor)

Manages its own manifest for each configuration and restores
the original when done (or on failure).

Usage:
    cd tutortrace_dataset_and_benchmark
    python3 benchmark/run_all_benchmarks.py
"""

import os
import sys
import yaml
import json
import shutil
import subprocess

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'manifest.yaml')
RESULTS_DIR = os.path.join(ROOT_DIR, 'benchmark', 'all_results')

TASKS = {
    'next_behavioral_state': True,
    'error_imminence': True,
    'query_imminence': True,
    'post_query_improvement': True,
}

ALL_DEPLOYMENTS = {
    'deployment_1': 'raw_telemetry/deployment_1.json',
    'deployment_2': 'raw_telemetry/deployment_2.json',
    'deployment_3': 'raw_telemetry/deployment_3.json',
    'deployment_4': 'raw_telemetry/deployment_4.json',
    'deployment_5': 'raw_telemetry/deployment_5.json',
    'deployment_6': 'raw_telemetry/deployment_6.json',
    'deployment_7': 'raw_telemetry/deployment_7.json',
    'deployment_8': 'raw_telemetry/deployment_8.json',
}


def build_manifest(train_deps, test_deps):
    """Build a manifest dict with specified train/test splits."""
    deployments = {}
    for name, path in ALL_DEPLOYMENTS.items():
        if name in train_deps:
            deployments[name] = {
                'raw_telemetry': path,
                'split': 'train',
                'enabled': True,
            }
        elif name in test_deps:
            deployments[name] = {
                'raw_telemetry': path,
                'split': 'test',
                'enabled': True,
            }
        else:
            deployments[name] = {
                'raw_telemetry': path,
                'split': 'test',
                'enabled': False,
            }
    return {'deployments': deployments, 'tasks': TASKS}


def write_manifest(manifest_dict):
    """Write manifest to disk."""
    with open(MANIFEST_PATH, 'w') as f:
        yaml.dump(manifest_dict, f, default_flow_style=False)


def run_prepare_and_benchmark(config_name):
    """Run prepare_data.py --force and run_benchmark.py, saving output."""
    print(f"\n  Preparing data...")
    result = subprocess.run(
        ['python3', os.path.join(ROOT_DIR, 'prepare_data.py'), '--force'],
        cwd=ROOT_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR in prepare_data.py: {result.stderr[:500]}")
        return False

    print(f"  Running benchmark...")
    result = subprocess.run(
        ['python3', os.path.join(ROOT_DIR, 'benchmark', 'run_benchmark.py')],
        cwd=ROOT_DIR, capture_output=True, text=True,
    )

    # Save output
    output_path = os.path.join(RESULTS_DIR, f'{config_name}_output.txt')
    with open(output_path, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write('\n\nSTDERR:\n')
            f.write(result.stderr)

    # Save results.json
    src_results = os.path.join(ROOT_DIR, 'benchmark', 'results.json')
    if os.path.exists(src_results):
        dst_results = os.path.join(RESULTS_DIR, f'{config_name}_results.json')
        shutil.copy2(src_results, dst_results)

    if result.returncode != 0:
        print(f"  WARNING: benchmark returned code {result.returncode}")
        return False

    print(f"  ✓ {config_name} complete")
    return True


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Backup original manifest
    backup_path = MANIFEST_PATH + '.backup'
    shutil.copy2(MANIFEST_PATH, backup_path)

    try:
        # ══════════════════════════════════════════════════════
        #  1. SETUP A: D1 → D2 (main results)
        # ══════════════════════════════════════════════════════

        print("=" * 60)
        print("  CONFIG 1: SETUP A (D1 → D2)")
        print("=" * 60)

        manifest = build_manifest(
            train_deps=['deployment_1'],
            test_deps=['deployment_2'],
        )
        write_manifest(manifest)
        run_prepare_and_benchmark('setup_a')

        # ══════════════════════════════════════════════════════
        #  2. PER-DEPLOYMENT: D1 → each individually
        # ══════════════════════════════════════════════════════

        test_deployments = [f'deployment_{i}' for i in range(2, 9)]

        for test_dep in test_deployments:
            dep_num = test_dep.split('_')[1]
            config_name = f'per_deploy_d{dep_num}'

            print(f"\n{'=' * 60}")
            print(f"  CONFIG: D1 → {test_dep}")
            print(f"{'=' * 60}")

            manifest = build_manifest(
                train_deps=['deployment_1'],
                test_deps=[test_dep],
            )
            write_manifest(manifest)
            run_prepare_and_benchmark(config_name)

        # ══════════════════════════════════════════════════════
        #  3. SETUP B: D1+D3+D4 → D2+D5 (multi-instructor)
        # ══════════════════════════════════════════════════════

        print(f"\n{'=' * 60}")
        print(f"  CONFIG: SETUP B (D1+D3+D4 → D2+D5)")
        print(f"{'=' * 60}")

        manifest = build_manifest(
            train_deps=['deployment_1', 'deployment_3', 'deployment_4'],
            test_deps=['deployment_2', 'deployment_5'],
        )
        write_manifest(manifest)
        run_prepare_and_benchmark('setup_b')

        # ══════════════════════════════════════════════════════
        #  DONE
        # ══════════════════════════════════════════════════════

        print(f"\n{'=' * 60}")
        print(f"  ALL CONFIGURATIONS COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Results saved to: {RESULTS_DIR}/")
        print(f"  Files:")
        for f in sorted(os.listdir(RESULTS_DIR)):
            print(f"    {f}")

    finally:
        # Always restore original manifest
        shutil.copy2(backup_path, MANIFEST_PATH)
        os.remove(backup_path)
        print(f"\n  Manifest restored to original state.")


if __name__ == '__main__':
    main()