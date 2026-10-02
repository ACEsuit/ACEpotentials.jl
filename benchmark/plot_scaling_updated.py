#!/usr/bin/env python3
"""
Plot MPI scaling comparison between Old ACE, ETACE (latest), and ML-PACE.

Benchmark: TiAl B2 10x10x10 (2000 atoms), order=3, degree=10, 100 MD steps
Updated: 2026-01-04 with latest ETACE deployment
"""
import matplotlib.pyplot as plt
import numpy as np

# Process counts tested
np_array = np.array([1, 2, 4, 8])

# Old ACE (Julia compiled library, v0.9+ export)
# Times from benchmark/results/oldace_np*.log
oldace_times = np.array([30.46, 16.00, 8.42, 4.58])  # seconds
oldace_perf = np.array([0.284, 0.540, 1.026, 1.888])  # ns/day

# ETACE (Latest - Jan 2026)
# Times from benchmark/results/etace_new_np*.log
etace_times = np.array([34.99, 19.42, 10.07, 5.71])  # seconds
etace_perf = np.array([0.247, 0.445, 0.858, 1.513])  # ns/day

# ML-PACE (v0.6.9 .yace export)
# Times from benchmark/results/pace_np*.log
pace_times = np.array([16.35, 8.22, 4.29, 2.80])  # seconds
pace_perf = np.array([0.529, 1.051, 2.015, 3.081])  # ns/day

# Calculate speedup relative to 1 process
oldace_speedup = oldace_times[0] / oldace_times
etace_speedup = etace_times[0] / etace_times
pace_speedup = pace_times[0] / pace_times
ideal_speedup = np_array

# Calculate parallel efficiency
oldace_efficiency = oldace_speedup / np_array * 100
etace_efficiency = etace_speedup / np_array * 100
pace_efficiency = pace_speedup / np_array * 100

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Wall time vs processes
ax1 = axes[0, 0]
ax1.plot(np_array, oldace_times, 'o-', label='Old ACE (juliac)', linewidth=2, markersize=8, color='C0')
ax1.plot(np_array, etace_times, 's-', label='ETACE (latest)', linewidth=2, markersize=8, color='C2')
ax1.plot(np_array, pace_times, '^-', label='ML-PACE', linewidth=2, markersize=8, color='C1')
ax1.set_xlabel('MPI Processes', fontsize=11)
ax1.set_ylabel('Wall Time (s)', fontsize=11)
ax1.set_title('Wall Time vs. MPI Processes', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(np_array)

# Plot 2: Performance (ns/day)
ax2 = axes[0, 1]
ax2.plot(np_array, oldace_perf, 'o-', label='Old ACE (juliac)', linewidth=2, markersize=8, color='C0')
ax2.plot(np_array, etace_perf, 's-', label='ETACE (latest)', linewidth=2, markersize=8, color='C2')
ax2.plot(np_array, pace_perf, '^-', label='ML-PACE', linewidth=2, markersize=8, color='C1')
ax2.set_xlabel('MPI Processes', fontsize=11)
ax2.set_ylabel('Performance (ns/day)', fontsize=11)
ax2.set_title('Performance vs. MPI Processes', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(np_array)

# Plot 3: Parallel Speedup
ax3 = axes[1, 0]
ax3.plot(np_array, oldace_speedup, 'o-', label='Old ACE (juliac)', linewidth=2, markersize=8, color='C0')
ax3.plot(np_array, etace_speedup, 's-', label='ETACE (latest)', linewidth=2, markersize=8, color='C2')
ax3.plot(np_array, pace_speedup, '^-', label='ML-PACE', linewidth=2, markersize=8, color='C1')
ax3.plot(np_array, ideal_speedup, 'k--', label='Ideal', linewidth=1)
ax3.set_xlabel('MPI Processes', fontsize=11)
ax3.set_ylabel('Speedup (T1/Tn)', fontsize=11)
ax3.set_title('Parallel Speedup', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(np_array)

# Plot 4: Parallel Efficiency
ax4 = axes[1, 1]
ax4.plot(np_array, oldace_efficiency, 'o-', label='Old ACE (juliac)', linewidth=2, markersize=8, color='C0')
ax4.plot(np_array, etace_efficiency, 's-', label='ETACE (latest)', linewidth=2, markersize=8, color='C2')
ax4.plot(np_array, pace_efficiency, '^-', label='ML-PACE', linewidth=2, markersize=8, color='C1')
ax4.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=1)
ax4.set_xlabel('MPI Processes', fontsize=11)
ax4.set_ylabel('Efficiency (%)', fontsize=11)
ax4.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(np_array)
ax4.set_ylim(0, 110)

plt.suptitle('MPI Scaling: Old ACE vs ETACE (Latest) vs ML-PACE\nTiAl B2 10x10x10 (2000 atoms), order=3, degree=10',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/results/scaling_comparison_updated.png', dpi=150)
plt.savefig('/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/results/scaling_comparison_updated.pdf')
print("Saved plots to benchmark/results/scaling_comparison_updated.png and .pdf")

# Print summary table
print("\n" + "="*100)
print("MPI SCALING BENCHMARK SUMMARY - Updated Jan 2026")
print("="*100)
print(f"System: TiAl B2, 10x10x10 supercell (2000 atoms)")
print(f"Model: order=3, totaldegree=10, rcut=5.5")
print(f"Benchmark: 100 MD steps with NVE at 300K")
print("="*100)

print(f"\n{'Procs':<6} {'Old ACE':<20} {'ETACE (latest)':<20} {'ML-PACE':<20} {'PACE/ETACE':<12}")
print(f"{'':<6} {'Time(s)':<10} {'ns/day':<10} {'Time(s)':<10} {'ns/day':<10} {'Time(s)':<10} {'ns/day':<10} {'Speedup':<12}")
print("-"*100)
for i, nproc in enumerate(np_array):
    ratio = pace_perf[i] / etace_perf[i]
    print(f"{nproc:<6} {oldace_times[i]:<10.2f} {oldace_perf[i]:<10.3f} "
          f"{etace_times[i]:<10.2f} {etace_perf[i]:<10.3f} "
          f"{pace_times[i]:<10.2f} {pace_perf[i]:<10.3f} {ratio:<12.2f}x")
print("="*100)

print("\n" + "-"*100)
print("PERFORMANCE ANALYSIS")
print("-"*100)
print(f"  Old ACE vs ETACE:   {oldace_perf[0]/etace_perf[0]:.2f}x faster at 1 core, {oldace_perf[-1]/etace_perf[-1]:.2f}x at 8 cores")
print(f"  ML-PACE vs ETACE:   {pace_perf[0]/etace_perf[0]:.2f}x faster at 1 core, {pace_perf[-1]/etace_perf[-1]:.2f}x at 8 cores")
print(f"  ML-PACE vs Old ACE: {pace_perf[0]/oldace_perf[0]:.2f}x faster at 1 core, {pace_perf[-1]/oldace_perf[-1]:.2f}x at 8 cores")
print("")
print("Key Findings:")
print(f"  1. ML-PACE is ~{pace_perf[0]/etace_perf[0]:.1f}x FASTER than ETACE at 1 core")
print(f"  2. ML-PACE is ~{pace_perf[-1]/etace_perf[-1]:.1f}x FASTER than ETACE at 8 cores")
print(f"  3. Old ACE is comparable to ETACE in performance")
print(f"  4. All potentials show excellent MPI scaling (70-80% efficiency at 8 cores)")
print("-"*100)

print(f"\nParallel scaling summary at 8 cores:")
print(f"  Old ACE speedup:  {oldace_speedup[-1]:.2f}x  (efficiency: {oldace_efficiency[-1]:.1f}%)")
print(f"  ETACE speedup:    {etace_speedup[-1]:.2f}x  (efficiency: {etace_efficiency[-1]:.1f}%)")
print(f"  ML-PACE speedup:  {pace_speedup[-1]:.2f}x  (efficiency: {pace_efficiency[-1]:.1f}%)")
print("\nConclusion: ML-PACE (v0.6.9) outperforms both Old ACE and ETACE exports.")
print("            ETACE and Old ACE show similar performance characteristics.")
print("            All three implementations show good parallel scaling efficiency.")
