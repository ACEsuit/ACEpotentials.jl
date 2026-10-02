#!/bin/bash
# Fair comparison benchmark script for four ACE export methods
# Run all benchmarks with varying MPI process counts

set -e

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$BENCHMARK_DIR/results"
LAMMPS_DIR="$BENCHMARK_DIR/lammps"
DEPLOY_DIR="$BENCHMARK_DIR/deployments"

# LAMMPS executable
LAMMPS="${LAMMPS:-/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp}"

# Check LAMMPS exists
if [ ! -f "$LAMMPS" ]; then
    echo "Error: LAMMPS not found at $LAMMPS"
    echo "Set LAMMPS environment variable to your LAMMPS executable"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Set up environment
export PATH="/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/bin:$PATH"
export LD_LIBRARY_PATH="/software/easybuild/software/GCCcore/14.3.0/lib64:$LD_LIBRARY_PATH"

# Add all library directories to LD_LIBRARY_PATH
for lib_dir in "$DEPLOY_DIR"/*/lib; do
    if [ -d "$lib_dir" ]; then
        export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
    fi
done
export LD_LIBRARY_PATH="/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/lib:$LD_LIBRARY_PATH"

# Plugin path (for juliac-compiled models)
PLUGIN_PATH="$BENCHMARK_DIR/../deployments/tial_ace/lammps/plugin/build/aceplugin.so"

# Library paths
OLDACE_LIB="$DEPLOY_DIR/oldace/lib/libace_fair_oldace.so"
ETACE_SPLINE_LIB="$DEPLOY_DIR/etace_spline/lib/libace_fair_etace_spline.so"
ETACE_POLY_LIB="$DEPLOY_DIR/etace_poly/lib/libace_fair_etace_poly.so"
MLPACE_YACE="$DEPLOY_DIR/mlpace/fair_mlpace.yace"
MLPACE_TABLE="$DEPLOY_DIR/mlpace/fair_mlpace_pairpot.table"

# Process counts to test
PROCS=(1 2 4 8)

# Benchmark methods
declare -A METHODS
METHODS["oldace"]="Old ACE (juliac)"
METHODS["etace_spline"]="ETACE Spline"
METHODS["etace_poly"]="ETACE Polynomial"
METHODS["mlpace"]="ML-PACE (native)"

echo "=============================================="
echo "Fair Benchmark: Four ACE Export Methods"
echo "=============================================="
echo "Date: $(date)"
echo "LAMMPS: $LAMMPS"
echo "Results directory: $RESULTS_DIR"
echo ""

# Function to run a benchmark
run_benchmark() {
    local method=$1
    local np=$2
    local input_file="$LAMMPS_DIR/in.fair_$method"
    local log_file="$RESULTS_DIR/fair_${method}_np${np}.log"

    echo "----------------------------------------------"
    echo "Running ${METHODS[$method]} with $np MPI process(es)..."
    echo "----------------------------------------------"

    # Create temporary input file with resolved paths
    local temp_input="/tmp/in.fair_${method}_$$"

    case $method in
        "oldace")
            if [ ! -f "$OLDACE_LIB" ]; then
                echo "Warning: $OLDACE_LIB not found, skipping"
                return 1
            fi
            sed -e "s|PLUGIN_PATH|$PLUGIN_PATH|g" \
                -e "s|OLDACE_LIB_PATH|$OLDACE_LIB|g" \
                "$input_file" > "$temp_input"
            ;;
        "etace_spline")
            if [ ! -f "$ETACE_SPLINE_LIB" ]; then
                echo "Warning: $ETACE_SPLINE_LIB not found, skipping"
                return 1
            fi
            sed -e "s|PLUGIN_PATH|$PLUGIN_PATH|g" \
                -e "s|ETACE_SPLINE_LIB_PATH|$ETACE_SPLINE_LIB|g" \
                "$input_file" > "$temp_input"
            ;;
        "etace_poly")
            if [ ! -f "$ETACE_POLY_LIB" ]; then
                echo "Warning: $ETACE_POLY_LIB not found, skipping"
                return 1
            fi
            sed -e "s|PLUGIN_PATH|$PLUGIN_PATH|g" \
                -e "s|ETACE_POLY_LIB_PATH|$ETACE_POLY_LIB|g" \
                "$input_file" > "$temp_input"
            ;;
        "mlpace")
            if [ ! -f "$MLPACE_YACE" ]; then
                echo "Warning: $MLPACE_YACE not found, skipping"
                return 1
            fi
            sed -e "s|MLPACE_YACE_PATH|$MLPACE_YACE|g" \
                -e "s|MLPACE_TABLE_PATH|$MLPACE_TABLE|g" \
                "$input_file" > "$temp_input"
            ;;
    esac

    # Run benchmark
    if [ $np -eq 1 ]; then
        $LAMMPS -in "$temp_input" 2>&1 | tee "$log_file"
    else
        mpirun -np $np $LAMMPS -in "$temp_input" 2>&1 | tee "$log_file"
    fi

    # Clean up
    rm -f "$temp_input"

    # Extract key metrics
    echo ""
    echo "Key metrics:"
    grep -E "(Loop time|Performance|Pair)" "$log_file" | tail -3
    echo ""
}

# Check what deployments exist
echo "Checking deployments..."
for method in oldace etace_spline etace_poly mlpace; do
    case $method in
        "oldace") lib="$OLDACE_LIB" ;;
        "etace_spline") lib="$ETACE_SPLINE_LIB" ;;
        "etace_poly") lib="$ETACE_POLY_LIB" ;;
        "mlpace") lib="$MLPACE_YACE" ;;
    esac
    if [ -f "$lib" ]; then
        echo "  [OK] ${METHODS[$method]}: $lib"
    else
        echo "  [MISSING] ${METHODS[$method]}: $lib"
    fi
done
echo ""

# Ask user to continue
read -p "Continue with available deployments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run all benchmarks
for method in oldace mlpace etace_spline etace_poly; do
    for np in "${PROCS[@]}"; do
        run_benchmark "$method" "$np" || echo "Skipped $method (np=$np)"
    done
done

echo "=============================================="
echo "Benchmarks complete!"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Run analyze_results.jl to generate comparison report"
