#!/usr/bin/env python3
"""
Analyze HPC hybrid MPI+OpenMP scaling benchmark results.
"""
import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

def parse_log(filepath):
    """Extract metrics from LAMMPS log file."""
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Performance (ns/day)
    perf_match = re.findall(r'Performance:\s+([\d.]+)\s+ns/day', content)

    # Loop time
    loop_match = re.findall(r'Loop time of ([\d.]+) on \d+ procs for 100 steps', content)

    # CPU utilization
    cpu_match = re.findall(r'([\d.]+)% CPU use with (\d+) MPI tasks x (\d+) OpenMP', content)

    # Atom count
    atom_match = re.search(r'Created (\d+) atoms', content)

    if perf_match and loop_match:
        result = {
            'ns_day': float(perf_match[-1]),
            'loop_time': float(loop_match[-1]),
            'atoms': int(atom_match.group(1)) if atom_match else None,
        }
        if cpu_match:
            result['cpu_pct'] = float(cpu_match[-1][0])
            result['mpi'] = int(cpu_match[-1][1])
            result['omp'] = int(cpu_match[-1][2])
        return result
    return None


def load_results():
    """Load all benchmark results."""
    results = defaultdict(lambda: defaultdict(dict))

    for logfile in RESULTS_DIR.glob("*.log"):
        # Parse filename: {approach}_{size}_np{N}_omp{M}.log
        match = re.match(r'(\w+)_(\w+)_np(\d+)_omp(\d+)\.log', logfile.name)
        if not match:
            continue

        approach, size, np, omp = match.groups()
        np, omp = int(np), int(omp)

        data = parse_log(logfile)
        if data:
            results[approach][size][(np, omp)] = data
            print(f"Loaded: {logfile.name} -> {data['ns_day']:.3f} ns/day")

    return results


def save_csv(results):
    """Save results to CSV."""
    csv_path = RESULTS_DIR / "scaling_summary.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['approach', 'size', 'mpi', 'omp', 'total_cores',
                        'ns_day', 'loop_time', 'cpu_pct', 'atoms'])

        for approach in results:
            for size in results[approach]:
                for (np, omp), data in results[approach][size].items():
                    writer.writerow([
                        approach, size, np, omp, np*omp,
                        data.get('ns_day', ''),
                        data.get('loop_time', ''),
                        data.get('cpu_pct', ''),
                        data.get('atoms', '')
                    ])

    print(f"\nSaved: {csv_path}")


def plot_results(results):
    """Generate scaling plots."""

    sizes = ['small', 'medium', 'large']
    size_atoms = {'small': 2000, 'medium': 16000, 'large': 54000}

    # Check what data we have
    has_juliac = 'juliac' in results and any(results['juliac'].values())
    has_pace = 'pace' in results and any(results['pace'].values())

    if not has_juliac:
        print("No juliac results found - plotting pace only")

    # Get all configs tested
    all_configs = set()
    for approach in results:
        for size in results[approach]:
            all_configs.update(results[approach][size].keys())

    if not all_configs:
        print("No results to plot!")
        return

    configs = sorted(all_configs, key=lambda x: (-x[0], x[1]))  # Sort by MPI desc
    config_labels = [f"{np}×{omp}" for np, omp in configs]

    # Create figure
    n_sizes = len([s for s in sizes if any(
        s in results.get(a, {}) for a in results
    )])

    if n_sizes == 0:
        print("No valid size data found!")
        return

    fig, axes = plt.subplots(n_sizes, 2, figsize=(14, 5*n_sizes))
    if n_sizes == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for size in sizes:
        # Check if we have data for this size
        juliac_data = results.get('juliac', {}).get(size, {})
        pace_data = results.get('pace', {}).get(size, {})

        if not juliac_data and not pace_data:
            continue

        natoms = size_atoms.get(size, '?')

        # Left plot: Absolute performance
        ax1 = axes[row, 0]
        x = np.arange(len(configs))
        width = 0.35

        if juliac_data:
            juliac_perf = [juliac_data.get(c, {}).get('ns_day', 0) for c in configs]
            ax1.bar(x - width/2, juliac_perf, width, label='juliac', color='steelblue')

        if pace_data:
            pace_perf = [pace_data.get(c, {}).get('ns_day', 0) for c in configs]
            ax1.bar(x + width/2, pace_perf, width, label='ML-PACE', color='coral', alpha=0.7)

        ax1.set_xlabel('Configuration (MPI × OMP)')
        ax1.set_ylabel('Performance (ns/day)')
        ax1.set_title(f'{size.upper()}: {natoms:,} atoms - Absolute Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(config_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Right plot: juliac relative performance + CPU utilization
        ax2 = axes[row, 1]

        if juliac_data:
            # Find pure MPI baseline (highest MPI count)
            pure_mpi_config = max([c for c in configs if c in juliac_data], key=lambda c: c[0])
            baseline_perf = juliac_data[pure_mpi_config]['ns_day']

            rel_perf = [(juliac_data.get(c, {}).get('ns_day', 0) / baseline_perf * 100)
                       if c in juliac_data else 0 for c in configs]

            colors = ['forestgreen' if p >= 95 else 'orange' if p >= 85 else 'tomato'
                     for p in rel_perf]
            bars = ax2.bar(x, rel_perf, width*1.5, color=colors, edgecolor='black')
            ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1.5)
            ax2.axhline(y=90, color='orange', linestyle=':', linewidth=1, alpha=0.5)

            # Add CPU% annotations
            for i, c in enumerate(configs):
                if c in juliac_data and 'cpu_pct' in juliac_data[c]:
                    cpu = juliac_data[c]['cpu_pct']
                    ax2.annotate(f'{cpu:.0f}%', (i, rel_perf[i]),
                               textcoords="offset points", xytext=(0, 5),
                               ha='center', fontsize=8, color='blue')

            ax2.set_ylabel('Relative Performance (%) + CPU%')
            ax2.set_title(f'{size.upper()}: juliac Scaling (vs pure MPI baseline)')
        else:
            ax2.text(0.5, 0.5, 'No juliac data', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f'{size.upper()}: juliac Scaling')

        ax2.set_xlabel('Configuration (MPI × OMP)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(config_labels, rotation=45, ha='right')
        ax2.set_ylim(0, 120)
        ax2.grid(True, alpha=0.3, axis='y')

        row += 1

    plt.suptitle('HPC Hybrid MPI+OpenMP Scaling Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = RESULTS_DIR / "scaling_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")

    # Also save PDF
    plt.savefig(RESULTS_DIR / "scaling_plots.pdf", bbox_inches='tight')


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("HPC SCALING BENCHMARK SUMMARY")
    print("="*80)

    for size in ['small', 'medium', 'large']:
        juliac_data = results.get('juliac', {}).get(size, {})
        pace_data = results.get('pace', {}).get(size, {})

        if not juliac_data and not pace_data:
            continue

        print(f"\n--- {size.upper()} ---")
        print(f"{'Config':<12} {'juliac (ns/day)':<18} {'ML-PACE (ns/day)':<18} {'CPU%':<10}")
        print("-"*60)

        all_configs = set(juliac_data.keys()) | set(pace_data.keys())
        for config in sorted(all_configs, key=lambda x: (-x[0], x[1])):
            np, omp = config
            j = juliac_data.get(config, {})
            p = pace_data.get(config, {})

            j_perf = f"{j.get('ns_day', 0):.3f}" if j else "N/A"
            p_perf = f"{p.get('ns_day', 0):.3f}" if p else "N/A"
            cpu = f"{j.get('cpu_pct', 0):.0f}%" if j.get('cpu_pct') else "N/A"

            print(f"{np}×{omp:<10} {j_perf:<18} {p_perf:<18} {cpu:<10}")

    print("="*80)


if __name__ == '__main__':
    print("Loading HPC scaling benchmark results...")
    print(f"Results directory: {RESULTS_DIR}\n")

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Run the benchmark first: ./run_scaling.sh")
        exit(1)

    results = load_results()

    if any(results.values()):
        print_summary(results)
        save_csv(results)
        plot_results(results)
    else:
        print("No results found. Run the benchmark first.")
