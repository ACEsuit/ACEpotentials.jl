# TiAl (Ti-Al, correlation order 4) LARGE reference model: the fit produced by
# export/bench/fit_tial_order4.jl, reloaded from its saved parameters.  DO NOT refit here.
#
# Why a second reference model at all: the Cantor fixture is order 3 with 1348 many-body basis
# functions per species; the performance tasks need a model whose many-body term dominates, and
# this one has 2369 per species at order 4.  It is also the model the matched `pair_style pace`
# comparator bench_parity/tial_o4_pace.ace is sized against.
#
# The model-construction block below is a verbatim copy of `tial_model` in
# export/bench/fit_tial_order4.jl; the saved `ps` only fits that exact spec, so any edit there
# shows up either as a Lux parameter-shape error or as a silently wrong energy.
#
# Exports (same shape as the Cantor fixture):
#   load_tial_fixture()          -> (; model, ps, st, E0s, stacked, held, rcut, elements, hypers)
#   tial_substacks(fx)           -> (onebody_calc, pair_calc, ace_calc)   [by model type name]
#   tial_mb_stack(fx)            -> StackedCalculator (E0 + many-body, pair dropped)
#   tial_spline_stack(fx; Nspl[, with_pair]) -> StackedCalculator
#   load_tial_held()             -> Vector of the 10 export-check configurations
#
# `held` is the 10 BULK configurations of the TiAl_tutorial dataset (5 x 128 atoms + 5 x 54
# atoms) -- the `check_idx` set of fit_tial_order4.jl.  Unlike the Cantor `held`, these are
# NOT a statistical held-out set: the tutorial dataset contains only 14 bulk cells in total and
# most of them had to stay in the training set.  Their role here is geometric coverage for the
# 1e-12 export gate, which compares exported code against the Julia calculator on the same
# geometry and is therefore indifferent to in-sample/out-of-sample.  The genuinely held-out
# configurations (indices `fx.hypers` ... see `held_idx` in the .jld2) are used only by the fit
# script's error table.

using ACEpotentials
using ACEpotentials.Models, ACEpotentials.ETModels
using AtomsBase, AtomsCalculators, Unitful, StaticArrays, LinearAlgebra
using Lux, LuxCore, JLD2, Random

# These three are also defined by cantor_fixture.jl; `const` redefinition with an identical
# value is a no-op in Julia, so including both fixtures in one process is safe.
const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
const NL = ACEpotentials.NeighbourLists     # not a dependency of export/Project.toml

const TIAL_REPO   = normpath(joinpath(@__DIR__, "..", "..", ".."))
const TIAL_PARAMS = joinpath(TIAL_REPO, "bench_parity", "tial_o4_params.jld2")
const TIAL_PACE   = joinpath(TIAL_REPO, "bench_parity", "tial_o4_pace.ace")

const TIAL_ELEMENTS  = (:Ti, :Al)
const TIAL_ORDER     = 4
const TIAL_RCUT      = 5.5
const TIAL_MAX_LEVEL = 11
const TIAL_PAIR_MAXN = 20
const TIAL_WL        = 1.5

const _TIAL_CACHE = Ref{Any}(nothing)

"""
    load_tial_fixture(; refresh = false)

Rebuild the TiAl order-4 model from `bench_parity/tial_o4_params.jld2` and return

    (; model, ps, st, E0s, stacked, held, rcut, elements, hypers)

* `model`   -- `ACEpotentials.Models.ACEModel`, built exactly as the fit built it
* `ps, st`  -- the fitted Lux parameters/state loaded from the .jld2
* `E0s`     -- `Dict{Symbol,Float64}` of per-element reference energies
* `stacked` -- `ETModels.StackedCalculator` of three `WrappedSiteCalculator`s in the order
               `(ETOneBody, ETPairModel, ETACE)` returned by `ETM.convert2et_full`
* `held`    -- the 10 bulk export-check configurations (see the note at the top of this file)
* `rcut`    -- 5.5 Å
* `hypers`  -- the NamedTuple of hyper-parameters recorded by the fit script

Memoised; pass `refresh = true` to rebuild.
"""
function load_tial_fixture(; refresh::Bool = false)
    if !refresh && _TIAL_CACHE[] !== nothing
        return _TIAL_CACHE[]
    end
    isfile(TIAL_PARAMS) || error("""
        $TIAL_PARAMS not found.  Run the fit first (it is not checked in -- ~15 min):
            cd $TIAL_REPO && mkdir -p bench_parity
            nohup julia --project=export export/bench/fit_tial_order4.jl \\
                  > bench_parity/fit_tial.log 2>&1 &""")
    ps, st, E0s, check_idx, hypers =
        JLD2.load(TIAL_PARAMS, "ps", "st", "E0s", "check_idx", "hypers")

    # --- verbatim copy of `tial_model` in export/bench/fit_tial_order4.jl ------------------
    # MUST stay identical to the fit; `E0s` is not just an initialisation, `convert2et_full`
    # reads `model.Vref.E0` to build the ETOneBody term of the exported stack.
    NZ = length(TIAL_ELEMENTS)
    r0 = M._default_rin0cuts(TIAL_ELEMENTS)
    rin0cuts = SMatrix{NZ, NZ}([(rin = 0.0, r0 = r0[i, j].r0, rcut = TIAL_RCUT)
                                for i in 1:NZ, j in 1:NZ])
    level = M.TotalDegree(1.0 * NZ, 1 / TIAL_WL)
    model = M.ace_model(; elements = TIAL_ELEMENTS, order = TIAL_ORDER, Ytype = :solid,
                          level = level, max_level = TIAL_MAX_LEVEL,
                          pair_maxn = TIAL_PAIR_MAXN, rin0cuts = rin0cuts,
                          init_WB = :zeros, init_Wpair = :onehot, init_Wradial = :onehot,
                          pair_learnable = true, E0s = E0s)
    # --- end verbatim copy -----------------------------------------------------------------

    stacked = ETM.convert2et_full(model, ps, st)    # (ETOneBody, ETPairModel, ETACE)
    held = load_tial_held(check_idx)
    fx = (; model, ps, st, E0s, stacked, held, rcut = TIAL_RCUT,
            elements = TIAL_ELEMENTS, hypers)
    _TIAL_CACHE[] = fx
    return fx
end

"""
    load_tial_held([check_idx]) -> Vector{<:AbstractSystem}

The 10 bulk configurations of `TiAl_tutorial` the export accuracy gate runs on.  `check_idx`
defaults to the index list recorded in the .jld2 by the fit script, so the gate set cannot
drift away from the one the fit recorded.
"""
function load_tial_held(check_idx = JLD2.load(TIAL_PARAMS, "check_idx"))
    data, _, _ = ACEpotentials.example_dataset("TiAl_tutorial")
    return data[check_idx]
end

"""
    tial_substacks(fx) -> (onebody_calc, pair_calc, ace_calc)

Locate the three sub-calculators of `fx.stacked` **by the name of their wrapped model type**
rather than by position (same rationale, and same failure message, as `cantor_substacks`).
"""
function tial_substacks(fx)
    calcs = fx.stacked.calcs
    names = [string(nameof(typeof(c.model))) for c in calcs]
    @assert length(calcs) == 3 "expected 3 stacked components from convert2et_full, got $(length(calcs)): $names"
    i1   = findfirst(n -> occursin("OneBody", n), names)
    ipr  = findfirst(n -> occursin("Pair", n), names)
    iace = findfirst(n -> n == "ETACE", names)
    @assert i1 !== nothing && ipr !== nothing && iace !== nothing """
        could not identify the (OneBody, Pair, ACE) sub-calculators of fx.stacked.
        Wrapped model types were: $names
        (found OneBody at $i1, Pair at $ipr, ACE at $iace).
        ETModels.convert2et_full has probably changed its component set or naming --
        update tial_substacks in export/test/fixtures/tial_fixture.jl."""
    return (calcs[i1], calcs[ipr], calcs[iace])
end

"""
    tial_mb_stack(fx) -> StackedCalculator

The `E0 + many-body` stack (pair term dropped).  Since Task 1 the generator emits the pair
term in both radial modes, so this is NOT the reference for an exported model -- it exists only
to measure how large the pair contribution is.
"""
function tial_mb_stack(fx)
    onebody_calc, _, ace_calc = tial_substacks(fx)
    return ETM.StackedCalculator((onebody_calc, ace_calc))
end

"""
    tial_spline_stack(fx; Nspl, with_pair = true) -> StackedCalculator

The reference a `:hermite_spline` export must be compared against: `(ETOneBody, ETPairModel,
splinified ETACE)`.  `splinify` replaces only the many-body radial basis; the pair term is
carried over unchanged.  `with_pair = false` drops it and is not a valid export reference.
"""
function tial_spline_stack(fx; Nspl::Integer, with_pair::Bool = true)
    onebody_calc, pair_calc, ace_calc = tial_substacks(fx)
    m = ETM.splinify(ace_calc.model, ace_calc.ps, ace_calc.st; Nspl = Nspl)
    p, s = LuxCore.setup(MersenneTwister(1), m)
    p.readout.W .= ace_calc.ps.readout.W
    spl_calc = ETM.ETACEPotential(m, p, s, fx.rcut)
    return with_pair ? ETM.StackedCalculator((onebody_calc, pair_calc, spl_calc)) :
                       ETM.StackedCalculator((onebody_calc, spl_calc))
end
