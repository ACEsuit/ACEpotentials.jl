#!/bin/bash
# Hybrid MPI+OpenMP scaling benchmark
# Tests different combinations of MPI ranks × OpenMP threads = 8 cores total
#
# Purpose: Determine if hybrid parallelism improves juliac performance
# relative to ML-PACE, or if both scale similarly.

set -e

# Load required modules
module load GCC/14.3.0
module load OpenMPI/5.0.8

# Configuration
TOTAL_CORES=8
BENCHMARK_DIR="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark"
LAMMPS="/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"
RESULTS_DIR="$BENCHMARK_DIR/results/hybrid"

# Set up library paths
export LD_LIBRARY_PATH="/software/easybuild/software/GCCcore/14.3.0/lib64:$BENCHMARK_DIR/deployments/tial_ace/lib:/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/lib:$LD_LIBRARY_PATH"

# Create results directory
mkdir -p "$RESULTS_DIR"

# MPI × OMP configurations (product = 8 cores)
declare -a CONFIGS=(
    "8:1"   # 8 MPI ranks × 1 thread (pure MPI baseline)
    "4:2"   # 4 MPI ranks × 2 threads
    "2:4"   # 2 MPI ranks × 4 threads
    "1:8"   # 1 MPI rank × 8 threads (pure OpenMP)
)

echo "=============================================="
echo "Hybrid MPI+OpenMP Scaling Benchmark"
echo "=============================================="
echo "Total cores: $TOTAL_CORES"
echo "LAMMPS: $LAMMPS"
echo "Date: $(date)"
echo ""

run_benchmark() {
    local approach=$1      # "juliac" or "pace"
    local input_file=$2
    local label=$3

    echo ""
    echo "############################################"
    echo "# Testing: $label"
    echo "############################################"

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r np omp <<< "$config"

        echo ""
        echo "--- $np MPI × $omp OMP = $((np * omp)) cores ---"

        LOG_FILE="$RESULTS_DIR/${approach}_np${np}_omp${omp}.log"

        # Set OpenMP threads
        export OMP_NUM_THREADS=$omp
        export OMP_PROC_BIND=spread
        export OMP_PLACES=threads

        # Run LAMMPS
        # Note: Some MPI implementations need explicit thread binding
        mpirun -np $np \
            --map-by socket:PE=$omp \
            --bind-to core \
            $LAMMPS -in "$input_file" \
            2>&1 | tee "$LOG_FILE"

        # Extract key metrics
        echo ""
        echo "Results ($np MPI × $omp OMP):"
        grep -E "(Loop time|Performance|Pair.*\|)" "$LOG_FILE" | tail -3
    done
}

# Run juliac benchmarks
echo ""
echo "=============================================="
echo "JULIAC EXPORT BENCHMARKS"
echo "=============================================="
run_benchmark "juliac" "$BENCHMARK_DIR/lammps/in.tial_benchmark" "juliac (new ACE plugin)"

# Run ML-PACE benchmarks
echo ""
echo "=============================================="
echo "ML-PACE BENCHMARKS"
echo "=============================================="
run_benchmark "pace" "$BENCHMARK_DIR/lammps/in.tial_pace" "ML-PACE (v0.6.9)"

# Summary
echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "To analyze results, run:"
echo "  python3 $BENCHMARK_DIR/plot_hybrid_scaling.py"
