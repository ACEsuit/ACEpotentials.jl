# PkgBenchmark suite for ACEpotentials force/energy evaluation.
#
# Used by PkgBenchmark.jl / BenchmarkCI.jl to compare a PR against its base
# branch on the same runner (the comparison cancels most hardware noise).
#
# Local run:
#   julia --project=benchmark -e 'using PkgBenchmark; \
#       r = benchmarkpkg("ACEpotentials"; script="benchmark/benchmarks.jl"); \
#       export_markdown(stdout, r)'
#
# The suite is intentionally small (2 system sizes, both backends) to keep CI
# runtime bounded. The `et` (autograd→analytic) force entries are the ones that
# guard against re-introducing the autograd regression.

using BenchmarkTools

include("common.jl")

const SUITE = BenchmarkGroup()

# Build the parameter-matched calculators once.
const _setup = build_model_and_calcs()
const _ace   = _setup.ace_calc
const _et    = _setup.et_calc

SUITE["energy"] = BenchmarkGroup()
SUITE["forces"] = BenchmarkGroup()

for cfg in CI_SIZE_CONFIGS
   sys = make_system(cfg)
   nat = length(sys)
   key = "$(nat)atoms"

   # warm up so the recorded samples exclude first-call compilation
   AtomsCalculators.potential_energy(sys, _ace)
   AtomsCalculators.potential_energy(sys, _et)
   AtomsCalculators.forces(sys, _ace)
   AtomsCalculators.forces(sys, _et)

   SUITE["energy"]["classic_$key"] =
      @benchmarkable AtomsCalculators.potential_energy($sys, $_ace)
   SUITE["energy"]["et_$key"] =
      @benchmarkable AtomsCalculators.potential_energy($sys, $_et)
   SUITE["forces"]["classic_$key"] =
      @benchmarkable AtomsCalculators.forces($sys, $_ace)
   SUITE["forces"]["et_$key"] =
      @benchmarkable AtomsCalculators.forces($sys, $_et)
end
