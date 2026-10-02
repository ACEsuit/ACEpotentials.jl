#!/bin/bash
# Full MPI scaling benchmark comparing old ACE, ETACE, and ML-PACE
# Run with: bash benchmark/lammps/run_full_scaling.sh

set -e

# Load MPI modules
source /etc/profile.d/modules.sh
module load GCC/14.3.0
module load OpenMPI/5.0.8

# Set paths
BENCHMARK_DIR="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark"
LAMMPS="/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"
RESULTS_DIR="$BENCHMARK_DIR/results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Set up library paths
export LD_LIBRARY_PATH="/software/easybuild/software/GCCcore/14.3.0/lib64:$BENCHMARK_DIR/deployments/tial_ace/lib:$BENCHMARK_DIR/deployments/tial_etace/lib:/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/lib:$LD_LIBRARY_PATH"

echo "=============================================="
echo "Full MPI Scaling Benchmark"
echo "Comparing: Old ACE, ETACE Hermite, ETACE Poly, ML-PACE"
echo "=============================================="
echo ""
echo "LAMMPS: $LAMMPS"
echo "Results: $RESULTS_DIR"
echo ""

# Process counts to test (up to 8 cores as requested)
PROCS="1 2 4 8"

# ==============================================
# Benchmark 1: Old ACE (Julia compiled library)
# ==============================================
echo ""
echo "############################################"
echo "# OLD ACE (Julia compiled library)         #"
echo "############################################"
INPUT="$BENCHMARK_DIR/lammps/in.tial_benchmark"

for np in $PROCS; do
    echo ""
    echo "--- Old ACE with $np MPI processes ---"
    LOG_FILE="$RESULTS_DIR/oldace_np${np}.log"

    OMP_NUM_THREADS=1 mpirun -np $np $LAMMPS -in $INPUT 2>&1 | tee "$LOG_FILE"

    echo ""
    grep -E "(Loop time|Performance)" "$LOG_FILE" | tail -2
done

# ==============================================
# Benchmark 2: ETACE (Hermite spline export)
# ==============================================
echo ""
echo "############################################"
echo "# ETACE (Hermite cubic spline, exact)      #"
echo "############################################"
INPUT="$BENCHMARK_DIR/lammps/in.tial_etace_codegen"

for np in $PROCS; do
    echo ""
    echo "--- ETACE Hermite with $np MPI processes ---"
    LOG_FILE="$RESULTS_DIR/etace_hermite_np${np}.log"

    OMP_NUM_THREADS=1 mpirun -np $np $LAMMPS -in $INPUT 2>&1 | tee "$LOG_FILE"

    echo ""
    grep -E "(Loop time|Performance)" "$LOG_FILE" | tail -2
done

# ==============================================
# Benchmark 3: ETACE (polynomial export)
# ==============================================
echo ""
echo "############################################"
echo "# ETACE (polynomial basis)                 #"
echo "############################################"
INPUT="$BENCHMARK_DIR/lammps/in.tial_etace_poly"

for np in $PROCS; do
    echo ""
    echo "--- ETACE Polynomial with $np MPI processes ---"
    LOG_FILE="$RESULTS_DIR/etace_poly_np${np}.log"

    OMP_NUM_THREADS=1 mpirun -np $np $LAMMPS -in $INPUT 2>&1 | tee "$LOG_FILE"

    echo ""
    grep -E "(Loop time|Performance)" "$LOG_FILE" | tail -2
done

# ==============================================
# Benchmark 4: ML-PACE (v0.6.9 .yace export)
# ==============================================
echo ""
echo "############################################"
echo "# ML-PACE (v0.6.9 .yace export)            #"
echo "############################################"
INPUT="$BENCHMARK_DIR/lammps/in.tial_pace"

for np in $PROCS; do
    echo ""
    echo "--- ML-PACE with $np MPI processes ---"
    LOG_FILE="$RESULTS_DIR/pace_np${np}.log"

    OMP_NUM_THREADS=1 mpirun -np $np $LAMMPS -in $INPUT 2>&1 | tee "$LOG_FILE"

    echo ""
    grep -E "(Loop time|Performance)" "$LOG_FILE" | tail -2
done

# ==============================================
# Summary
# ==============================================
echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE - SUMMARY"
echo "=============================================="
echo ""
echo "Process counts tested: $PROCS"
echo ""

echo "OLD ACE results:"
for np in $PROCS; do
    LOG_FILE="$RESULTS_DIR/oldace_np${np}.log"
    if [ -f "$LOG_FILE" ]; then
        TIME=$(grep "Loop time" "$LOG_FILE" | awk '{print $4}')
        PERF=$(grep "Performance" "$LOG_FILE" | awk '{print $2}')
        echo "  np=$np: ${TIME}s, ${PERF} ns/day"
    fi
done

echo ""
echo "ETACE Hermite results:"
for np in $PROCS; do
    LOG_FILE="$RESULTS_DIR/etace_hermite_np${np}.log"
    if [ -f "$LOG_FILE" ]; then
        TIME=$(grep "Loop time" "$LOG_FILE" | awk '{print $4}')
        PERF=$(grep "Performance" "$LOG_FILE" | awk '{print $2}')
        echo "  np=$np: ${TIME}s, ${PERF} ns/day"
    fi
done

echo ""
echo "ETACE Polynomial results:"
for np in $PROCS; do
    LOG_FILE="$RESULTS_DIR/etace_poly_np${np}.log"
    if [ -f "$LOG_FILE" ]; then
        TIME=$(grep "Loop time" "$LOG_FILE" | awk '{print $4}')
        PERF=$(grep "Performance" "$LOG_FILE" | awk '{print $2}')
        echo "  np=$np: ${TIME}s, ${PERF} ns/day"
    fi
done

echo ""
echo "ML-PACE results:"
for np in $PROCS; do
    LOG_FILE="$RESULTS_DIR/pace_np${np}.log"
    if [ -f "$LOG_FILE" ]; then
        TIME=$(grep "Loop time" "$LOG_FILE" | awk '{print $4}')
        PERF=$(grep "Performance" "$LOG_FILE" | awk '{print $2}')
        echo "  np=$np: ${TIME}s, ${PERF} ns/day"
    fi
done

echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "=============================================="
