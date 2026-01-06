# Performance Analysis: juliac Export vs ML-PACE

## Executive Summary

The juliac export approach is approximately **2x slower** than ML-PACE (v0.6.9) for equivalent ACE models. This gap is **architectural**, not due to memory allocation or micro-optimization opportunities. Critically, **both approaches scale equally well in parallel** - the 2x performance ratio remains constant across all process counts.

## Benchmark Results

**System**: TiAl B2, 10x10x10 supercell (2000 atoms), order=3, totaldegree=10, rcut=5.5

| Processes | juliac Time | juliac ns/day | ML-PACE Time | ML-PACE ns/day | Ratio | Abs Gap |
|-----------|-------------|---------------|--------------|----------------|-------|---------|
| 1 | 33.92s | 0.255 | 16.01s | 0.540 | 2.12x | 0.29 |
| 4 | 8.45s | 1.023 | 4.28s | 2.019 | 1.97x | 1.00 |
| 8 | 4.58s | 1.885 | 2.30s | 3.759 | 1.99x | 1.87 |

**Parallel scaling:**
- juliac: 7.4x speedup at 8 cores (92.6% efficiency)
- ML-PACE: 7.0x speedup at 8 cores (87.0% efficiency)

## Understanding the "Widening Gap"

**Important clarification:** On a linear-scale performance plot, the gap between juliac and ML-PACE appears to widen with more processes. This is a **visual artifact**, not a scaling deficiency.

Three ways to view the same data:

| Metric | 1 → 8 Processes | Interpretation |
|--------|-----------------|----------------|
| **Performance ratio** | 2.12x → 1.99x | Slightly *narrowing* (juliac catching up) |
| **Absolute gap** | 0.29 → 1.87 ns/day | Widening (misleading on linear plots) |
| **Parallel speedup** | juliac 7.4x, ML-PACE 7.0x | Both scale excellently |

**Why the absolute gap grows:** When both approaches scale by ~7x, the initial gap also scales by ~7x. If A=0.25 and B=0.54 at 1 core, then at 8 cores A≈1.9 and B≈3.8. The ratio (B/A) stays ~2x, but the difference (B-A) grows from 0.3 to 1.9.

**The log-scale plot** shows this clearly: the lines are parallel, indicating constant relative performance. The ratio plot confirms the gap stays at ~2x across all process counts.

## Root Cause Analysis

### What Was Investigated

1. **Memory allocation hypothesis** - Investigated whether per-atom allocations (~11KB) were the bottleneck
   - Implemented pre-allocated workspace buffers
   - Result: **No improvement** (actually 18-26% slower)
   - Conclusion: Julia's allocator is efficient; allocation is not the bottleneck

2. **View/dispatch overhead** - Tested using `@view` vs full arrays
   - `AbstractMatrix` dispatch adds overhead
   - SubArray indirection slows hot loops
   - Result: Workspace with views was **slower** than allocating version

3. **SIMD/bounds checking** - Added `@inbounds` and `@simd` annotations
   - Minor improvements but not significant
   - Limited by loop structure (complex operations in inner loops)

### The Real Bottleneck

The performance gap is due to **fundamental architectural differences**:

| Aspect | juliac Export | ML-PACE |
|--------|--------------|---------|
| Evaluation pattern | Per-atom, per-neighbor loops | Batch + cache-optimized |
| Basis evaluation | Scalar function calls | Vectorized with SIMD |
| Memory layout | Julia arrays (general purpose) | Custom cache-aligned |
| Tensor contraction | EquivariantTensors (general) | Hand-tuned sparse ops |

## Why Lux Migration Will Help

The planned migration to fully Lux-based models (PR #305, `co/etback` branch) will change the architecture to:

```
Current (per-atom):              Lux-based (batched):
for each atom:                   G = ETGraph(edges)
  for each neighbor:             r = map(transform, G.edge_data)  # all edges
    Rnl_j = eval(r_j)           Rnl = rbasis(r)                   # batched
    Ylm_j = eval(r̂_j)           Ylm = ybasis(r̂)                   # batched
  A = pool(Rnl, Ylm)            B = SparseACElayer(Rnl, Ylm)     # fused
  ...                           E = sum(B)
```

Key improvements from Lux migration:
1. **Vectorized evaluation** - All edges processed together
2. **GPU acceleration** - KernelAbstractions enables GPU execution
3. **Automatic differentiation** - Zygote handles gradients
4. **Better cache utilization** - Contiguous edge data in 3D arrays

## Recommendations

### Short-term (before Lux migration)
- **Do not pursue further micro-optimizations** - The bottleneck is architectural
- The current code is clean and correct; keep it maintainable
- Focus development effort on the Lux migration

### For Lux migration
- Use `mlip.jl` example from EquivariantTensors as template
- Ensure `juliac --trim` compatibility with KernelAbstractions (may need CPU-only path)
- Consider keeping C interface similar for LAMMPS compatibility

### Future benchmarking
- Re-run this benchmark after Lux migration
- Compare CPU (KernelAbstractions) vs GPU performance
- Target: Match or exceed ML-PACE on CPU

## Hybrid MPI+OpenMP Benchmark Results

### Key Finding: ML-PACE Does NOT Support OpenMP

Investigation of the ML-PACE source code reveals it has **no OpenMP support**:

```cpp
// pair_pace.cpp line 175 - Simple sequential loop, no #pragma omp
for (ii = 0; ii < inum; ii++) {
    // ... all work done serially
}
```

Searching the entire ML-PACE codebase (`/tmp/lammps-user-pace/ML-PACE/`) finds zero OpenMP pragmas.

### CPU Utilization Proves This

| Config | juliac CPU% | ML-PACE CPU% | Interpretation |
|--------|-------------|--------------|----------------|
| 1×8 | **677.8%** | 99.8% | juliac uses 7 threads; ML-PACE uses 1 |
| 2×4 | **379.1%** | 99.7% | juliac uses 4 threads; ML-PACE uses 1 |
| 4×2 | **194.3%** | 99.8% | juliac uses 2 threads; ML-PACE uses 1 |
| 8×1 | 99.2% | 99.7% | Both use 1 thread per rank |

When running `1 MPI × 8 OMP` with ML-PACE:
- LAMMPS allocates 8 threads
- `pair_pace` ignores them - has no OpenMP code
- **7 of 8 cores sit completely idle**
- Performance drops to 1/8 of what pure MPI would achieve

### Raw Benchmark Data

| Config | juliac (ns/day) | ML-PACE (ns/day) | Notes |
|--------|-----------------|------------------|-------|
| 8×1 (pure MPI) | 1.838 | 3.715 | **Valid comparison**: ML-PACE 2x faster |
| 4×2 | 1.808 | 2.002 | ML-PACE wastes 1 thread/rank |
| 2×4 | 1.768 | 1.036 | ML-PACE wastes 3 threads/rank |
| 1×8 (pure OMP) | 1.599 | 0.547 | ML-PACE wastes 7 threads |

### Correct Interpretation

The hybrid benchmark results are **not a fair comparison** for configs other than 8×1:
- ML-PACE is designed for **pure MPI only** (or Kokkos/GPU via `pace/kk`)
- juliac actually implements OpenMP threading over atoms
- Comparing them with OpenMP threads allocated but unused by ML-PACE is misleading

### What ML-PACE Supports

1. **MPI parallelism** - domain decomposition (works well)
2. **Kokkos/GPU** - `pair_style pace/kk` (**NOT compatible with ACEpotentials.jl exports** - see below)
3. **NO OpenMP** - `pair_style pace` is purely sequential within each MPI rank

### pace/kk Does NOT Work with ACEpotentials.jl Exports

Investigation of the ML-PACE source code reveals a fundamental incompatibility:

**Class hierarchy:**
```
AbstractRadialBasis (base class)
├── ACERadialFunctions      (pace/kk supported)
└── ACEjlRadialFunctions    (ACEpotentials.jl exports)
```

**The problem:** `pair_pace_kokkos.cpp` line 255 requires `ACERadialFunctions`:
```cpp
ACERadialFunctions* radial_functions = dynamic_cast<ACERadialFunctions*>(...);
if (radial_functions == nullptr)
    error->all(FLERR,"Chosen radial basis style not supported by pair style pace/kk");
```

ACEpotentials.jl exports use `radbasename: "ACE.jl"` which creates `ACEjlRadialFunctions`.
Since this is NOT a subclass of `ACERadialFunctions`, the `dynamic_cast` fails → **pace/kk errors out**.

**Verified in our benchmark's .yace file:**
```yaml
bonds:
  [0, 0]:
    radbasename: "ACE.jl"    # Triggers ACEjlRadialFunctions, incompatible with pace/kk
```

**Implication:** For GPU acceleration of ACEpotentials.jl models, the Lux migration with KernelAbstractions is the only viable path. ML-PACE's GPU support (pace/kk) is only for potentials fitted with the Python pacemaker toolkit, not ACEpotentials.jl

### Valid Conclusions

| Scenario | juliac | ML-PACE (.yace export) |
|----------|--------|------------------------|
| Pure MPI | ✓ Works (~2x slower) | ✓ Works (fastest for CPU) |
| Hybrid MPI+OpenMP | ✓ Works (uses threads) | ✗ Threads wasted |
| GPU (Kokkos) | Pending Lux migration | ✗ pace/kk incompatible with ACEpotentials.jl |

**Bottom line for ACEpotentials.jl users:**
- **CPU (pure MPI):** Export to .yace and use ML-PACE for best performance
- **CPU (hybrid):** Use juliac plugin (only option that uses OpenMP)
- **GPU:** Not currently available; awaiting Lux migration

### juliac's OpenMP Implementation

The juliac/ACE plugin does use OpenMP threading. With `1 MPI × 8 OMP`:
- CPU utilization: 677.8% (using ~7 of 8 threads)
- Performance: 1.599 ns/day (only 13% slower than 8×1)
- Threads parallelize over atoms within the MPI domain

This is a genuine advantage of the juliac approach for hybrid parallelism scenarios.

### To Reproduce

```bash
cd benchmark/lammps
./run_hybrid_scaling.sh

cd ..
python3 plot_hybrid_scaling.py
```

Results saved in `results/hybrid/`.

## Files Modified During Investigation

All changes were **reverted** as they did not improve performance:
- `export/src/export_ace_model.jl` - Workspace code (reverted)

## Baseline for Future Comparison

```
juliac export baseline (2024-12-11):
- TiAl B2 2000 atoms, order=3, degree=10
- Single core: 33.92s, 0.255 ns/day
- 8 cores: 4.58s, 1.885 ns/day
- Speedup: 7.4x (92.6% efficiency)

ML-PACE v0.6.9 baseline:
- Same system
- Single core: 16.01s, 0.540 ns/day
- 8 cores: 2.30s, 3.759 ns/day
- Speedup: 7.0x (87.0% efficiency)

Performance ratio: ~2x constant across all process counts
```
