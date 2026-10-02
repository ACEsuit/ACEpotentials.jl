#!/usr/bin/env python3
"""
Analyze and plot hybrid MPI+OpenMP scaling benchmark results.

Parses LAMMPS log files from the hybrid benchmark and generates comparison plots.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

RESULTS_DIR = Path("/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/results/hybrid")

# Configurations tested (MPI × OMP)
CONFIGS = [
    (8, 1),  # Pure MPI
    (4, 2),
    (2, 4),
    (1, 8),  # Pure OpenMP
]

def parse_lammps_log(filepath):
    """Extract performance metrics from LAMMPS log file."""
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Find the MD run performance line (not minimization)
    # Format: "Performance: X.XXX ns/day, Y.YYY hours/ns, Z.ZZZ timesteps/s"
    perf_matches = re.findall(r'Performance:\s+([\d.]+)\s+ns/day', content)

    # Find loop time for MD run
    # Format: "Loop time of X.XXX on N procs for 100 steps"
    loop_matches = re.findall(r'Loop time of ([\d.]+) on \d+ procs for 100 steps', content)

    # Find Pair time from timing breakdown
    pair_matches = re.findall(r'Pair\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|', content)

    if perf_matches and loop_matches:
        return {
            'ns_day': float(perf_matches[-1]),  # Last match = MD run
            'loop_time': float(loop_matches[-1]),
            'pair_time': float(pair_matches[-1]) if pair_matches else None,
        }
    return None


def load_all_results():
    """Load results for all configurations and approaches."""
    results = {'juliac': {}, 'pace': {}}

    for approach in ['juliac', 'pace']:
        for np, omp in CONFIGS:
            filename = f"{approach}_np{np}_omp{omp}.log"
            filepath = RESULTS_DIR / filename
            data = parse_lammps_log(filepath)
            if data:
                results[approach][(np, omp)] = data
                print(f"Loaded {filename}: {data['ns_day']:.3f} ns/day, {data['loop_time']:.2f}s")
            else:
                print(f"Missing or invalid: {filename}")

    return results


def plot_results(results):
    """Generate comparison plots."""
    if not results['juliac'] and not results['pace']:
        print("No results to plot!")
        return

    # Extract data for plotting
    config_labels = [f"{np}×{omp}" for np, omp in CONFIGS]

    juliac_perf = [results['juliac'].get(c, {}).get('ns_day', np.nan) for c in CONFIGS]
    pace_perf = [results['pace'].get(c, {}).get('ns_day', np.nan) for c in CONFIGS]

    juliac_time = [results['juliac'].get(c, {}).get('loop_time', np.nan) for c in CONFIGS]
    pace_time = [results['pace'].get(c, {}).get('loop_time', np.nan) for c in CONFIGS]

    # Calculate ratios where both exist
    ratios = []
    for c in CONFIGS:
        j = results['juliac'].get(c, {}).get('ns_day')
        p = results['pace'].get(c, {}).get('ns_day')
        if j and p:
            ratios.append(p / j)
        else:
            ratios.append(np.nan)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = np.arange(len(CONFIGS))
    width = 0.35

    # Plot 1: Performance (ns/day)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, juliac_perf, width, label='juliac', color='C0')
    bars2 = ax1.bar(x + width/2, pace_perf, width, label='ML-PACE', color='C1')
    ax1.set_xlabel('Configuration (MPI × OMP)')
    ax1.set_ylabel('Performance (ns/day)')
    ax1.set_title('Performance vs Hybrid Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Wall time
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, juliac_time, width, label='juliac', color='C0')
    ax2.bar(x + width/2, pace_time, width, label='ML-PACE', color='C1')
    ax2.set_xlabel('Configuration (MPI × OMP)')
    ax2.set_ylabel('Wall Time (s)')
    ax2.set_title('Wall Time vs Hybrid Configuration')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Performance ratio (ML-PACE / juliac)
    ax3 = axes[1, 0]
    ax3.bar(x, ratios, width*1.5, color='C2')
    ax3.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, label='2x reference')
    ax3.set_xlabel('Configuration (MPI × OMP)')
    ax3.set_ylabel('Performance Ratio (ML-PACE / juliac)')
    ax3.set_title('Performance Ratio vs Hybrid Configuration\n(constant ≈ 2x means equal scaling)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_labels)
    ax3.set_ylim(0, 3)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    # Annotate values
    for i, v in enumerate(ratios):
        if not np.isnan(v):
            ax3.annotate(f'{v:.2f}x', (i, v), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=10)

    # Plot 4: Speedup relative to pure MPI (8×1)
    ax4 = axes[1, 1]
    juliac_base = results['juliac'].get((8, 1), {}).get('ns_day', 1)
    pace_base = results['pace'].get((8, 1), {}).get('ns_day', 1)

    juliac_speedup = [p / juliac_base if not np.isnan(p) else np.nan for p in juliac_perf]
    pace_speedup = [p / pace_base if not np.isnan(p) else np.nan for p in pace_perf]

    ax4.bar(x - width/2, juliac_speedup, width, label='juliac', color='C0')
    ax4.bar(x + width/2, pace_speedup, width, label='ML-PACE', color='C1')
    ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('Configuration (MPI × OMP)')
    ax4.set_ylabel('Speedup vs Pure MPI (8×1)')
    ax4.set_title('Relative Performance vs Pure MPI Baseline')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Hybrid MPI+OpenMP Scaling: juliac vs ML-PACE\n'
                 'TiAl B2 10×10×10 (2000 atoms), 8 cores total',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = RESULTS_DIR / 'hybrid_scaling_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")

    # Also save PDF
    plt.savefig(RESULTS_DIR / 'hybrid_scaling_comparison.pdf')


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("HYBRID MPI+OPENMP SCALING SUMMARY")
    print("="*80)
    print(f"{'Config':<12} {'juliac':<22} {'ML-PACE':<22} {'Ratio':<10}")
    print(f"{'(MPI×OMP)':<12} {'Time(s)':<11} {'ns/day':<11} {'Time(s)':<11} {'ns/day':<11} {'PACE/juliac':<10}")
    print("-"*80)

    for config in CONFIGS:
        nprocs, omp = config
        j = results['juliac'].get(config, {})
        p = results['pace'].get(config, {})

        j_time = j.get('loop_time', float('nan'))
        j_perf = j.get('ns_day', float('nan'))
        p_time = p.get('loop_time', float('nan'))
        p_perf = p.get('ns_day', float('nan'))

        if j_perf and p_perf:
            ratio = p_perf / j_perf
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"

        print(f"{nprocs}×{omp:<10} {j_time:<11.2f} {j_perf:<11.3f} {p_time:<11.2f} {p_perf:<11.3f} {ratio_str:<10}")

    print("="*80)

    # Analysis
    print("\nKEY FINDINGS:")

    # Check if ratio stays constant
    ratios = []
    for c in CONFIGS:
        j = results['juliac'].get(c, {}).get('ns_day')
        p = results['pace'].get(c, {}).get('ns_day')
        if j and p:
            ratios.append(p / j)

    if ratios:
        ratio_mean = np.mean(np.array(ratios))
        ratio_std = np.std(np.array(ratios))
        print(f"  - Performance ratio across configs: {ratio_mean:.2f}x ± {ratio_std:.2f}")

        if ratio_std < 0.2:
            print("  - Ratio is STABLE: Both approaches scale similarly with hybrid parallelism")
            print("  - The 2x gap is architectural, not a parallelization issue")
        else:
            print("  - Ratio VARIES: One approach scales better with hybrid parallelism")

            # Find best config for each
            best_juliac = max(CONFIGS, key=lambda c: results['juliac'].get(c, {}).get('ns_day', 0))
            best_pace = max(CONFIGS, key=lambda c: results['pace'].get(c, {}).get('ns_day', 0))
            print(f"  - Best config for juliac: {best_juliac[0]}×{best_juliac[1]}")
            print(f"  - Best config for ML-PACE: {best_pace[0]}×{best_pace[1]}")


if __name__ == '__main__':
    print("Loading hybrid scaling benchmark results...")
    print(f"Results directory: {RESULTS_DIR}")
    print()

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Run the benchmark first: ./lammps/run_hybrid_scaling.sh")
        exit(1)

    results = load_all_results()

    if results['juliac'] or results['pace']:
        print_summary(results)
        plot_results(results)
    else:
        print("No results found. Run the benchmark first.")
