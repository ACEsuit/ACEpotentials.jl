#!/bin/bash
set -e

cd /home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/fair_comparison

export LD_LIBRARY_PATH="/software/easybuild/software/GCCcore/14.3.0/lib64:$(pwd)/deployments/oldace/lib:$(pwd)/deployments/etace_spline/lib:$(pwd)/deployments/etace_poly/lib:/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/lib:$LD_LIBRARY_PATH"

LAMMPS="/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"
PLUGIN="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/deployments/tial_ace/lammps/plugin/build/aceplugin.so"

mkdir -p results

echo "=============================================="
echo "Fair Benchmark: All ACE Export Methods"
echo "Date: $(date)"
echo "=============================================="

# Function to run benchmark
run_benchmark() {
    local method=$1
    local lib_or_yace=$2
    local extra_table=$3

    echo ""
    echo "=== Running $method ==="

    # Create temporary input file
    local input_file="lammps/in.fair_$method"
    local temp_input="/tmp/in.fair_${method}_$$"

    case $method in
        "oldace")
            sed -e "s|PLUGIN_PATH|$PLUGIN|g" \
                -e "s|OLDACE_LIB_PATH|$lib_or_yace|g" \
                "$input_file" > "$temp_input"
            ;;
        "etace_spline")
            sed -e "s|PLUGIN_PATH|$PLUGIN|g" \
                -e "s|ETACE_SPLINE_LIB_PATH|$lib_or_yace|g" \
                "$input_file" > "$temp_input"
            ;;
        "etace_poly")
            sed -e "s|PLUGIN_PATH|$PLUGIN|g" \
                -e "s|ETACE_POLY_LIB_PATH|$lib_or_yace|g" \
                "$input_file" > "$temp_input"
            ;;
        "mlpace")
            sed -e "s|MLPACE_YACE_PATH|$lib_or_yace|g" \
                -e "s|MLPACE_TABLE_PATH|$extra_table|g" \
                "$input_file" > "$temp_input"
            ;;
    esac

    # Run and save
    local log_file="results/fair_${method}_np1.log"
    $LAMMPS -in "$temp_input" 2>&1 | tee "$log_file"

    # Extract key metrics
    echo ""
    echo "Key metrics for $method:"
    grep -E "(Loop time|Performance|Pair)" "$log_file" | tail -3

    rm -f "$temp_input"
}

# Run Old ACE
run_benchmark "oldace" "$(pwd)/deployments/oldace/lib/libace_fair_oldace.so"

# Run ETACE Spline
run_benchmark "etace_spline" "$(pwd)/deployments/etace_spline/lib/libace_fair_etace_spline.so"

# Run ETACE Polynomial
run_benchmark "etace_poly" "$(pwd)/deployments/etace_poly/lib/libace_fair_etace_poly.so"

# Run ML-PACE
run_benchmark "mlpace" "$(pwd)/deployments/mlpace/fair_mlpace.yace" "$(pwd)/deployments/mlpace/fair_mlpace_pairpot.table"

echo ""
echo "=============================================="
echo "Benchmark Summary"
echo "=============================================="

for method in oldace etace_spline etace_poly mlpace; do
    log="results/fair_${method}_np1.log"
    if [ -f "$log" ]; then
        loop_time=$(grep "Loop time" "$log" | awk '{print $4}')
        perf=$(grep "Performance:" "$log" | awk '{print $2}')
        echo "$method: Loop time = ${loop_time}s, Performance = ${perf} ns/day"
    fi
done
