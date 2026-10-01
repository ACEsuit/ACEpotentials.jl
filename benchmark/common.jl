# Shared setup for the force-evaluation performance benchmarks.
#
# Single source of truth for:
#   - model construction (classic ACEModel + ET conversion)
#   - test systems of varying size
#   - the measurement helper used by both the reproduction script
#     (bench_forces_regression.jl) and the PkgBenchmark suite (benchmarks.jl).
#
# This deliberately mirrors benchmark/benchmark_full_model.jl so the numbers are
# comparable to the historical benchmark, while adding allocation capture and a
# parameter-matched classic-vs-ET pair built from the SAME (ps, st).

using ACEpotentials
const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

import EquivariantTensors as ET
import AtomsCalculators
using StaticArrays, Lux, Random, LuxCore, LinearAlgebra
using AtomsBase, AtomsBuilder, Unitful
using BenchmarkTools

# --- Model definition (kept small enough for CI, large enough to be meaningful) ---
const ELEMENTS = (:Si, :O)
const ORDER    = 2
const MAXLEVEL = 8
const MAXL     = 4

"""
    build_model_and_calcs(; rng)

Build a classic ACE model and its parameter-matched ET conversion from the SAME
`(model, ps, st)`, so the two calculators represent identical physics and any
timing difference is purely backend/implementation.

Returns `(; model, ps, st, ace_calc, et_calc, rcut)`.
"""
function build_model_and_calcs(; rng = Random.MersenneTwister(1234))
   rin0cuts = M._default_rin0cuts(ELEMENTS)
   rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)
   E0s = Dict(:Si => -158.54496821, :O => -2042.0330099956639)

   model = M.ace_model(; elements = ELEMENTS, order = ORDER,
                       Ytype = :solid, level = M.TotalDegree(),
                       max_level = MAXLEVEL, maxl = MAXL, pair_maxn = MAXLEVEL,
                       rin0cuts = rin0cuts,
                       init_WB = :glorot_normal, init_Wpair = :glorot_normal,
                       pair_learnable = true,   # required for ET conversion
                       E0s = E0s)

   ps, st = Lux.setup(rng, model)

   ace_calc = M.ACEPotential(model, ps, st)            # classic analytic backend
   et_calc  = ETM.convert2et_full(model, ps, st)       # ET (autograd) backend
   rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

   return (; model, ps, st, ace_calc, et_calc, rcut)
end

# --- Systems ---
# (nx, ny, nz) supercell repeats of cubic Si (8 atoms/cell), randomly half-Si/half-O.
const SIZE_CONFIGS = [
   (2, 2, 2),   #  64 atoms
   (3, 3, 2),   # 144 atoms
   (4, 4, 2),   # 256 atoms
   (4, 4, 4),   # 512 atoms
   (5, 5, 4),   # 800 atoms
]

# Smaller subset used by the CI PkgBenchmark suite to bound runtime.
const CI_SIZE_CONFIGS = [(2, 2, 2), (4, 4, 2)]   # 64, 256 atoms

"""
    make_system(cfg; rng)

Deterministic rattled Si/O supercell for the given `(nx,ny,nz)` config.
"""
function make_system(cfg; rng = Random.MersenneTwister(1234))
   sys = AtomsBuilder.bulk(:Si, cubic = true) * cfg
   rattle!(sys, 0.1u"Å")
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys
end

"""
    nedges(sys, rcut)

Number of directed edges in the interaction graph (for reporting).
"""
nedges(sys, rcut) = length(ET.Atoms.interaction_graph(sys, rcut * u"Å").edge_data)

"""
    measure(f)

Run `f()` once to warm up, then benchmark it. Returns `(; time_ns, memory, allocs)`
taken from the minimum sample (noise-resistant).
"""
function measure(f)
   f()  # warmup / compile
   b = @benchmark $f() samples = 5 evals = 3
   m = minimum(b)
   return (; time_ns = m.time, memory = m.memory, allocs = m.allocs)
end
