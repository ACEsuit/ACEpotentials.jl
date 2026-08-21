#!/bin/bash
#
# HPC Hybrid MPI+OpenMP Scaling Benchmark
# Tests juliac vs ML-PACE across different system sizes and parallelization strategies
#

set -e

#=============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR SYSTEM
#=============================================================================

# Total cores available (adjust to your allocation)
TOTAL_CORES=64

# LAMMPS executable path
LAMMPS="/path/to/lmp"

# Benchmark directory (where this script lives)
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load required modules (adjust for your system)
# module load gcc/14.3.0
# module load openmpi/5.0.8

#=============================================================================
# END CONFIGURATION
#=============================================================================

# Derived paths
INPUTS_DIR="$BENCH_DIR/inputs"
MODELS_DIR="$BENCH_DIR/models"
LIB_DIR="$BENCH_DIR/lib"
RESULTS_DIR="$BENCH_DIR/results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# System sizes to test
SIZES=("small" "medium" "large")

# Generate MPI×OMP configurations that multiply to TOTAL_CORES
# Format: "mpi:omp"
generate_configs() {
    local total=$1
    local configs=()

    # Find all factor pairs
    for ((mpi=1; mpi<=total; mpi++)); do
        if ((total % mpi == 0)); then
            omp=$((total / mpi))
            # Only include configs where OMP is power of 2 and <= 16
            if [[ $omp -le 16 ]] && (( (omp & (omp-1)) == 0 )); then
                configs+=("$mpi:$omp")
            fi
        fi
    done

    echo "${configs[@]}"
}

CONFIGS=($(generate_configs $TOTAL_CORES))

echo "=============================================================="
echo "HPC HYBRID MPI+OPENMP SCALING BENCHMARK"
echo "=============================================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Total cores: $TOTAL_CORES"
echo "Configurations: ${CONFIGS[*]}"
echo "System sizes: ${SIZES[*]}"
echo "LAMMPS: $LAMMPS"
echo "=============================================================="

# Check prerequisites
if [[ ! -x "$LAMMPS" ]]; then
    echo "ERROR: LAMMPS not found at $LAMMPS"
    echo "Please edit LAMMPS variable in this script"
    exit 1
fi

if [[ ! -f "$MODELS_DIR/tial_model.yace" ]]; then
    echo "ERROR: Model file not found: $MODELS_DIR/tial_model.yace"
    exit 1
fi

# Check for juliac library
JULIAC_LIB=$(ls "$LIB_DIR"/libace_*.so 2>/dev/null | head -1)
JULIAC_PLUGIN=$(ls "$LIB_DIR"/aceplugin.so 2>/dev/null | head -1)

HAS_JULIAC=false
if [[ -f "$JULIAC_LIB" && -f "$JULIAC_PLUGIN" ]]; then
    HAS_JULIAC=true
    echo "Found juliac library: $JULIAC_LIB"
fi

#=============================================================================
# BENCHMARK FUNCTIONS
#=============================================================================

run_pace() {
    local size=$1
    local np=$2
    local omp=$3
    local input="$INPUTS_DIR/in.tial_${size}"
    local log="$RESULTS_DIR/pace_${size}_np${np}_omp${omp}.log"

    echo "  ML-PACE: ${np} MPI × ${omp} OMP, size=$size"

    export OMP_NUM_THREADS=$omp
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads

    # Create temporary input with pair_style
    local tmp_input=$(mktemp)
    cat > "$tmp_input" << EOF
# ML-PACE potential
pair_style      hybrid/overlay pace table spline 5500
pair_coeff      * * pace ${MODELS_DIR}/tial_model.yace Ti Al
pair_coeff      1 1 table ${MODELS_DIR}/tial_model_pairpot.table Ti_Ti
pair_coeff      1 2 table ${MODELS_DIR}/tial_model_pairpot.table Al_Ti
pair_coeff      2 2 table ${MODELS_DIR}/tial_model_pairpot.table Al_Al

$(cat "$input")
EOF

    mpirun -np $np \
        --map-by socket:PE=$omp \
        --bind-to core \
        $LAMMPS -in "$tmp_input" > "$log" 2>&1 || true

    rm -f "$tmp_input"

    # Extract performance
    perf=$(grep "Performance:" "$log" | tail -1 | awk '{print $2}')
    echo "    -> ${perf:-FAILED} ns/day"
}

run_juliac() {
    local size=$1
    local np=$2
    local omp=$3
    local input="$INPUTS_DIR/in.tial_${size}"
    local log="$RESULTS_DIR/juliac_${size}_np${np}_omp${omp}.log"

    echo "  juliac: ${np} MPI × ${omp} OMP, size=$size"

    export OMP_NUM_THREADS=$omp
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads

    # Add library path
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"

    # Create temporary input with pair_style
    local tmp_input=$(mktemp)
    cat > "$tmp_input" << EOF
# juliac ACE plugin
plugin load ${JULIAC_PLUGIN}

pair_style      ace
pair_coeff      * * ${JULIAC_LIB} Ti Al

$(cat "$input")
EOF

    mpirun -np $np \
        --map-by socket:PE=$omp \
        --bind-to core \
        $LAMMPS -in "$tmp_input" > "$log" 2>&1 || true

    rm -f "$tmp_input"

    # Extract performance
    perf=$(grep "Performance:" "$log" | tail -1 | awk '{print $2}')
    cpu=$(grep "CPU use" "$log" | tail -1 | awk '{print $1}')
    echo "    -> ${perf:-FAILED} ns/day (CPU: ${cpu:-?})"
}

#=============================================================================
# MAIN BENCHMARK LOOP
#=============================================================================

echo ""
echo "Starting benchmarks..."
echo ""

for size in "${SIZES[@]}"; do
    echo "=============================================================="
    echo "SYSTEM SIZE: $size"
    echo "=============================================================="

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r np omp <<< "$config"
        echo ""
        echo "--- Configuration: ${np} MPI × ${omp} OMP ---"

        # Always run ML-PACE (for reference, even though it doesn't use OMP)
        run_pace "$size" "$np" "$omp"

        # Run juliac if available
        if $HAS_JULIAC; then
            run_juliac "$size" "$np" "$omp"
        fi
    done
done

echo ""
echo "=============================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================="
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "To analyze results, run:"
echo "  python3 $BENCH_DIR/analyze_results.py"
