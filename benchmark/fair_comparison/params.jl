# Unified model parameters for fair benchmark comparison
# All four methods (Old ACE, ML-PACE, ETACE Spline, ETACE Polynomial) use these parameters

module FairParams

using Unitful

export ELEMENTS, ORDER, TOTALDEGREE, MAXL, RCUT
export E0S, E0S_UNITFUL
export TRAIN_STRIDE, SOLVER_LAMBDA, PRIOR_P
export WEIGHTS
export BENCHMARK_STEPS, BENCHMARK_ATOMS, BENCHMARK_TEMP, BENCHMARK_SEED
export FAIR_DIR, DEPLOYMENTS_DIR, LAMMPS_DIR, RESULTS_DIR
export get_params, print_summary

# Model hyperparameters
const ELEMENTS = [:Ti, :Al]
const ORDER = 3
const TOTALDEGREE = 8
const MAXL = 4              # User-selected balanced value
const RCUT = 5.5            # Angstrom

# Reference energies (eV per atom)
const E0S = Dict(
    :Ti => -1586.0195,
    :Al => -105.5954
)
const E0S_UNITFUL = Dict(
    :Ti => -1586.0195u"eV",
    :Al => -105.5954u"eV"
)

# Training configuration
const TRAIN_STRIDE = 5      # Use every 5th config: data[1:5:end]
const SOLVER_LAMBDA = 1e-3
const PRIOR_P = 4           # algebraic_smoothness_prior(p=4)

# Fitting weights
const WEIGHTS = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0)
)

# LAMMPS benchmark settings
const BENCHMARK_STEPS = 100
const BENCHMARK_ATOMS = 2000  # 10x10x10 B2 supercell
const BENCHMARK_TEMP = 300.0  # K
const BENCHMARK_SEED = 12345

# Output paths
const FAIR_DIR = @__DIR__
const DEPLOYMENTS_DIR = joinpath(FAIR_DIR, "deployments")
const LAMMPS_DIR = joinpath(FAIR_DIR, "lammps")
const RESULTS_DIR = joinpath(FAIR_DIR, "results")

# Helper function to get NamedTuple of all params
function get_params()
    return (
        elements = ELEMENTS,
        order = ORDER,
        totaldegree = TOTALDEGREE,
        maxl = MAXL,
        rcut = RCUT,
        E0s = E0S,
        train_stride = TRAIN_STRIDE,
        solver_lambda = SOLVER_LAMBDA,
        prior_p = PRIOR_P,
        weights = WEIGHTS,
    )
end

# Print summary
function print_summary()
    println("=" ^ 60)
    println("Fair Benchmark Parameters")
    println("=" ^ 60)
    println("Elements:     $(join(ELEMENTS, ", "))")
    println("Order:        $ORDER")
    println("Totaldegree:  $TOTALDEGREE")
    println("Max L:        $MAXL")
    println("Cutoff:       $RCUT Å")
    println("E0s:          Ti=$(E0S[:Ti]) eV, Al=$(E0S[:Al]) eV")
    println("Training:     Every $(TRAIN_STRIDE)th configuration")
    println("Solver:       QR(lambda=$SOLVER_LAMBDA)")
    println("Prior:        algebraic_smoothness_prior(p=$PRIOR_P)")
    println("=" ^ 60)
end

end # module
