#!/bin/bash
set -e

cd /home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/fair_comparison

JULIAC="/home/eng/essswb/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/share/julia/juliac/juliac.jl"
PROJECT="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/export"

# Recompile ETACE spline without --trim=safe
echo "=== Compiling ETACE Spline... ==="
julia --project=$PROJECT $JULIAC \
    --output-lib deployments/etace_spline/lib/libace_fair_etace_spline.so \
    --experimental \
    deployments/etace_spline/fair_etace_spline_model.jl
echo "Done."

# Recompile ETACE poly without --trim=safe
echo "=== Compiling ETACE Polynomial... ==="
julia --project=$PROJECT $JULIAC \
    --output-lib deployments/etace_poly/lib/libace_fair_etace_poly.so \
    --experimental \
    deployments/etace_poly/fair_etace_poly_model.jl
echo "Done."

# Recompile Old ACE without --trim=safe
echo "=== Compiling Old ACE... ==="
julia --project=$PROJECT $JULIAC \
    --output-lib deployments/oldace/lib/libace_fair_oldace.so \
    --experimental \
    deployments/oldace/fair_oldace_model.jl
echo "Done."

# Check symbols
echo ""
echo "=== Checking exported symbols ==="
echo "ETACE Spline:"
nm -D deployments/etace_spline/lib/libace_fair_etace_spline.so | grep -c ace_ || echo "0"
echo "ETACE Poly:"
nm -D deployments/etace_poly/lib/libace_fair_etace_poly.so | grep -c ace_ || echo "0"
echo "Old ACE:"
nm -D deployments/oldace/lib/libace_fair_oldace.so | grep -c ace_ || echo "0"

echo ""
echo "=== File sizes ==="
ls -lh deployments/*/lib/*.so
