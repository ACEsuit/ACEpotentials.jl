# HPC Hybrid MPI+OpenMP Scaling Benchmark

Benchmarks juliac (ACEpotentials.jl compiled export) vs ML-PACE on larger systems
to determine optimal MPI×OpenMP configurations at scale.

## Contents

```
hpc_benchmark/
├── README.md                 # This file
├── run_scaling.sh           # Main benchmark script
├── analyze_results.py       # Analysis and plotting script
├── inputs/
│   ├── in.tial_small        # 10×10×10 = 2,000 atoms
│   ├── in.tial_medium       # 20×20×20 = 16,000 atoms
│   └── in.tial_large        # 30×30×30 = 54,000 atoms
├── models/
│   ├── tial_model.yace      # ML-PACE model (ACEpotentials.jl v0.6.9 export)
│   └── tial_model_pairpot.table
└── lib/
    └── (copy your compiled juliac library here)
```

## Prerequisites

1. **LAMMPS** compiled with:
   - ML-PACE package
   - OpenMP support
   - MPI support

2. **For juliac tests**: Copy your compiled ACE plugin to `lib/`:
   - `libace_*.so` - the compiled model
   - `aceplugin.so` - the LAMMPS plugin loader

3. **Environment modules** (adjust for your system):
   ```bash
   module load gcc/14.3.0    # Or your GCC version
   module load openmpi/5.0   # Or your MPI
   ```

## Quick Start

1. Edit `run_scaling.sh`:
   - Set `LAMMPS` to your LAMMPS executable path
   - Set `TOTAL_CORES` to match your allocation
   - Adjust module loads for your system

2. Run the benchmark:
   ```bash
   ./run_scaling.sh
   ```

3. Analyze results:
   ```bash
   python3 analyze_results.py
   ```

## Benchmark Configurations

The script tests multiple MPI×OMP combinations that multiply to your total cores:

| Total Cores | Configurations Tested |
|-------------|----------------------|
| 64 | 64×1, 32×2, 16×4, 8×8 |
| 128 | 128×1, 64×2, 32×4, 16×8 |
| 256 | 256×1, 128×2, 64×4, 32×8 |

## System Sizes

Three system sizes test different atoms-per-rank scenarios:

| Size | Atoms | At 64 ranks | At 8 ranks |
|------|-------|-------------|------------|
| Small | 2,000 | 31/rank | 250/rank |
| Medium | 16,000 | 250/rank | 2,000/rank |
| Large | 54,000 | 844/rank | 6,750/rank |

## Expected Outcomes

**If OpenMP helps at scale:**
- Hybrid configs (e.g., 16×4) should outperform pure MPI (64×1)
- Benefit should increase with system size (more atoms per domain)

**If pure MPI is always best:**
- 64×1 wins across all system sizes
- OpenMP overhead exceeds communication savings

## Output Files

Results are saved to `results/`:
- `{approach}_{size}_np{N}_omp{M}.log` - Raw LAMMPS output
- `scaling_summary.csv` - Parsed performance data
- `scaling_plots.png` - Visualization

## Notes

- juliac uses OpenMP; ML-PACE does NOT (included for reference only)
- Adjust `--map-by` and `--bind-to` options for your MPI and hardware
- For SLURM systems, see example job script in `submit_job.sh`
