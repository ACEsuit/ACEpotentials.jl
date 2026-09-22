#!/bin/bash
# Direct comparison: ML-PACE vs ETACE Spline
# Uses existing working ML-PACE deployment

set -e
cd /home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/fair_comparison

export LD_LIBRARY_PATH="/software/easybuild/software/GCCcore/14.3.0/lib64:$(pwd)/deployments/etace_spline/lib:/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/lib:$LD_LIBRARY_PATH"

LAMMPS="/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"
PLUGIN="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/deployments/tial_ace/lammps/plugin/build/aceplugin.so"
BENCHMARK_DIR="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark"

mkdir -p results

echo "=============================================="
echo "ML-PACE vs ETACE Spline Comparison"
echo "Date: $(date)"
echo "=============================================="

# Run ML-PACE (using existing working deployment)
echo ""
echo "=== Running ML-PACE (native C++) ==="
$LAMMPS -in $BENCHMARK_DIR/lammps/in.tial_pace 2>&1 | tee results/mlpace_comparison.log
echo ""
echo "ML-PACE Key Metrics:"
grep -E "(Loop time|Performance|Pair)" results/mlpace_comparison.log | tail -3

# Run ETACE Spline
echo ""
echo "=== Running ETACE Spline (juliac) ==="
cat > /tmp/in.etace_comparison << 'EOF'
# TiAl B2 structure benchmark - ETACE Spline
units           metal
atom_style      atomic
boundary        p p p

plugin          load PLUGIN_PATH

lattice         bcc 3.19
region          box block 0 10 0 10 0 10
create_box      2 box
create_atoms    1 box basis 1 1 basis 2 2

mass            1 47.867   # Ti
mass            2 26.982   # Al

pair_style      ace
pair_coeff      * * ETACE_LIB_PATH Ti Al

print           "Created B2 TiAl structure (ETACE Spline)"

minimize        1.0e-6 1.0e-8 100 1000
reset_timestep  0
velocity        all create 300.0 12345 dist gaussian
fix             1 all nve
timestep        0.001
timer           full
thermo          10
thermo_style    custom step pe ke etotal press temp
run             100
variable        final_pe equal pe
print           "Final potential energy: ${final_pe} eV"
EOF

ETACE_LIB="$(pwd)/deployments/etace_spline/lib/libace_fair_etace_spline.so"
sed -e "s|PLUGIN_PATH|$PLUGIN|g" -e "s|ETACE_LIB_PATH|$ETACE_LIB|g" /tmp/in.etace_comparison > /tmp/in.etace_run
$LAMMPS -in /tmp/in.etace_run 2>&1 | tee results/etace_spline_comparison.log
echo ""
echo "ETACE Spline Key Metrics:"
grep -E "(Loop time|Performance|Pair)" results/etace_spline_comparison.log | tail -3

# Extract and compare
echo ""
echo "=============================================="
echo "PERFORMANCE COMPARISON SUMMARY"
echo "=============================================="

MLPACE_PAIR=$(grep "^Pair" results/mlpace_comparison.log | tail -1 | awk '{print $3}')
ETACE_PAIR=$(grep "^Pair" results/etace_spline_comparison.log | tail -1 | awk '{print $3}')

MLPACE_LOOP=$(grep "Loop time" results/mlpace_comparison.log | tail -1 | awk '{print $4}')
ETACE_LOOP=$(grep "Loop time" results/etace_spline_comparison.log | tail -1 | awk '{print $4}')

echo "                    ML-PACE (C++)    ETACE Spline (juliac)"
echo "Pair time:          ${MLPACE_PAIR}s           ${ETACE_PAIR}s"
echo "Loop time:          ${MLPACE_LOOP}s           ${ETACE_LOOP}s"

# Calculate speedup using bc
if command -v bc &> /dev/null; then
    SPEEDUP=$(echo "scale=2; $MLPACE_PAIR / $ETACE_PAIR" | bc)
    echo ""
    echo "ETACE Spline speedup vs ML-PACE: ${SPEEDUP}x"
fi

rm -f /tmp/in.etace_comparison /tmp/in.etace_run
