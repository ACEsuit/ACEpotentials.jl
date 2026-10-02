#!/bin/bash
#SBATCH --job-name=ace_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err

#=============================================================================
# SLURM Job Script for HPC Hybrid Scaling Benchmark
#
# Adjust the #SBATCH directives above for your system:
#   --nodes          : Number of nodes
#   --ntasks-per-node: MPI tasks per node (set to max cores for flexibility)
#   --time           : Wall time limit
#   --partition      : Queue/partition name
#=============================================================================

# Load modules (adjust for your system)
module purge
module load gcc/14.3.0
module load openmpi/5.0.8
# module load cuda/12.0   # If testing GPU later

# Set paths
BENCH_DIR="$SLURM_SUBMIT_DIR"
cd "$BENCH_DIR"

# Configure the benchmark
export TOTAL_CORES=$SLURM_NTASKS

# Set LAMMPS path (adjust for your system)
export LAMMPS="/path/to/lmp"

# Optional: Set library paths if needed
# export LD_LIBRARY_PATH="$BENCH_DIR/lib:$LD_LIBRARY_PATH"

echo "=============================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "=============================================================="

# Run the benchmark
./run_scaling.sh

# Analyze results
python3 analyze_results.py

echo "Job complete: $(date)"
