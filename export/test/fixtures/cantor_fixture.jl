# Cantor (CrMnFeCoNi) reference model: the fit produced by verify_cantor/chain_cantor.jl,
# reloaded from its saved parameters.  DO NOT refit here (the fit takes ~43 min) and do NOT
# load verify_cantor/lsq_cantor.jld2 (1.7 GB design matrix).
#
# The model-construction block below is a verbatim copy of chain_cantor.jl lines 23-41; the
# saved `ps` only fits that exact spec, so any edit there shows up either as a Lux
# parameter-shape error or as a silently wrong energy.
#
# Exports:
#   load_cantor_fixture()        -> (; model, ps, st, E0s, stacked, held, held_xyz, rcut, elements)
#   cantor_substacks(fx)         -> (onebody_calc, pair_calc, ace_calc)   [by model type name]
#   cantor_mb_stack(fx)          -> StackedCalculator (E0 + many-body)
#   cantor_spline_stack(fx;Nspl) -> StackedCalculator (E0 + splinified ETACE)
#   load_cantor_reference(k)     -> (; natoms, E_a, E_amb, E_bmb, E_c50, E_c200, F_a, ...)
#   read_cantor_lammps_data(fn)  -> periodic_system
#
# `held` is the set of 10 *rotated* held-out geometries read from
# ~/si-ace/spike_yace/cantor/cantor_{1..10}.data.  Those are the configurations the 17-digit
# references verify_cantor/ref_{1..10}.txt, the matched .yace files and the LAMMPS inputs all
# use, so they are the canonical check set for this plan.  The un-rotated ExtXYZ originals
# (data[991:1000]) are available as `held_xyz` -- they are loaded lazily because reading the
# 1000-configuration .xyz is the single most expensive part of the fixture.

using ACEpotentials, ExtXYZ
using ACEpotentials.Models, ACEpotentials.ETModels
using AtomsBase, AtomsCalculators, Unitful, StaticArrays, LinearAlgebra
using Lux, LuxCore, JLD2, Random
using AtomsBase: ChemicalSpecies, atomic_number, position, cell_vectors, periodic_system
using Unitful: ustrip, @u_str

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
# NeighbourLists is a dependency of ACEpotentials but NOT of export/Project.toml, so it has to
# be reached through the ACEpotentials module rather than with a bare `using NeighbourLists`.
const NL = ACEpotentials.NeighbourLists

const CANTOR_REPO   = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CANTOR_VERIFY = joinpath(CANTOR_REPO, "verify_cantor")
const CANTOR_PARAMS = joinpath(CANTOR_VERIFY, "cantor_v010_params.jld2")
const CANTOR_XYZ    = "/home/eng/essswb/ACEpotentials-jax/cantor1k_b_mh1.xyz"
const CANTOR_YACE   = expanduser("~/si-ace/spike_yace/cantor")

const CANTOR_ELEMENTS = (:Cr, :Mn, :Fe, :Co, :Ni)
const CANTOR_RCUT = 6.25

"""
    read_cantor_lammps_data(fn) -> AbstractSystem

Read one of the rotated held-out LAMMPS data files.  Verbatim copy of
`chain_cantor.jl:99-111` (species index -> `CANTOR_ELEMENTS[type]`).
"""
function read_cantor_lammps_data(fn)
    L = readlines(fn)
    g(pat) = parse.(Float64, split(L[findfirst(l -> occursin(pat, l), L)])[1:end-length(split(pat))])
    xlo, xhi = g("xlo xhi"); ylo, yhi = g("ylo yhi"); zlo, zhi = g("zlo zhi"); xy, xz, yz = g("xy xz yz")
    box = [SVector(xhi - xlo, 0.0, 0.0), SVector(xy, yhi - ylo, 0.0), SVector(xz, yz, zhi - zlo)]
    ia = findfirst(l -> startswith(l, "Atoms"), L)
    atoms = Pair{Symbol, SVector{3,Float64}}[]
    for l in L[ia+2:end]
        isempty(strip(l)) && continue
        w = split(l)
        push!(atoms, CANTOR_ELEMENTS[parse(Int, w[2])] => SVector(parse.(Float64, w[3:5])...))
    end
    return periodic_system([e => x * u"Å" for (e, x) in atoms], box .* u"Å")
end

"Read the 10 rotated held-out geometries (the ones ref_k.txt / the .yace files refer to)."
load_cantor_held() =
    [read_cantor_lammps_data(joinpath(CANTOR_YACE, "cantor_$k.data")) for k in 1:10]

"The un-rotated ExtXYZ held-out configurations, data[991:1000].  Loads a 5 MB, 1000-config file."
load_cantor_held_xyz() = ExtXYZ.load(CANTOR_XYZ)[991:1000]

"""
    load_cantor_reference(k) -> NamedTuple

Parse `verify_cantor/ref_\$k.txt`, the 17-significant-digit reference written by
`chain_cantor.jl`.  Fields: `natoms`, energies `E_a, E_amb, E_bmb, E_c50, E_c200` and force
vectors `F_a, F_amb, F_bmb, F_c50, F_c200 :: Vector{SVector{3,Float64}}`, where
  a     = ACEModel, full (E0 + pair + many-body)
  amb   = ACEModel with Wpair = 0  (E0 + many-body)
  bmb   = ET StackedCalculator (ETOneBody + ETACE)
  c50   / c200 = E0 + splinified ETACE with Nspl = 50 / 200
"""
function load_cantor_reference(k::Integer)
    L = readlines(joinpath(CANTOR_VERIFY, "ref_$k.txt"))
    h = split(L[1])
    natoms = parse(Int, h[3])
    E = (; E_a = parse(Float64, h[5]), E_amb = parse(Float64, h[7]),
           E_bmb = parse(Float64, h[9]), E_c50 = parse(Float64, h[11]),
           E_c200 = parse(Float64, h[13]))
    F = [Vector{SVector{3,Float64}}(undef, natoms) for _ in 1:5]
    for (i, l) in enumerate(L[3:end])
        isempty(strip(l)) && continue
        w = parse.(Float64, split(l)[2:end])
        for c in 1:5
            F[c][i] = SVector{3,Float64}(w[3c-2], w[3c-1], w[3c])
        end
    end
    return (; natoms, E..., F_a = F[1], F_amb = F[2], F_bmb = F[3], F_c50 = F[4], F_c200 = F[5])
end

const _CANTOR_CACHE = Ref{Any}(nothing)

"""
    load_cantor_fixture(; refresh = false)

Rebuild the Cantor model from `verify_cantor/cantor_v010_params.jld2` and return

    (; model, ps, st, E0s, stacked, held, held_xyz, rcut, elements)

* `model`  -- `ACEpotentials.Models.ACEModel`, built exactly as the fit built it
* `ps, st` -- the fitted Lux parameters/state loaded from the .jld2
* `E0s`    -- `Dict{Symbol,Float64}` of per-element reference energies
* `stacked`-- `ETModels.StackedCalculator` of three `WrappedSiteCalculator`s, in the order
              `(ETOneBody, ETPairModel, ETACE)` returned by `ETM.convert2et_full`
* `held`   -- the 10 rotated held-out systems (see the note at the top of this file)
* `held_xyz` -- thunk-free: `nothing` until `load_cantor_held_xyz()` is called explicitly
* `rcut`   -- 6.25 Å

The result is memoised; pass `refresh = true` to rebuild.
"""
function load_cantor_fixture(; refresh::Bool = false)
    if !refresh && _CANTOR_CACHE[] !== nothing
        return _CANTOR_CACHE[]
    end
    ps, st, E0s = JLD2.load(CANTOR_PARAMS, "ps", "st", "E0s")

    # --- verbatim copy of chain_cantor.jl:23-41 -- MUST stay identical to the fit ----------
    elements = (:Cr, :Mn, :Fe, :Co, :Ni); NZ = 5
    rcut = 6.25; r0 = 2.54
    rin0cuts = SMatrix{NZ, NZ}([(rin = 0.0, r0 = r0, rcut = rcut) for i in 1:NZ, j in 1:NZ])
    level = M.TotalDegree(1.0 * NZ, 1 / 1.5)      # ace1_model's level weighting (wL = 1.5)
    model = M.ace_model(; elements = elements, order = 3, Ytype = :solid, level = level, max_level = 6,
                          pair_maxn = 30, rin0cuts = rin0cuts, init_WB = :zeros, init_Wpair = :onehot,
                          init_Wradial = :onehot, pair_learnable = true, E0s = E0s)
    # --- end verbatim copy -----------------------------------------------------------------

    stacked = ETM.convert2et_full(model, ps, st)    # (ETOneBody, ETPairModel, ETACE)
    held = load_cantor_held()
    fx = (; model, ps, st, E0s, stacked, held, held_xyz = nothing, rcut, elements)
    _CANTOR_CACHE[] = fx
    return fx
end

"""
    cantor_substacks(fx) -> (onebody_calc, pair_calc, ace_calc)

Locate the three sub-calculators of `fx.stacked` **by the name of their wrapped model type**
rather than by position, so that an upstream reordering of `StackedCalculator.calcs` fails
here with a readable message instead of surfacing as an opaque `nothing`-indexing error (or,
worse, as a silently wrong reference stack) inside a caller.

Single source of truth for this lookup: every helper that needs a sub-calculator goes through
it.  As of ACEpotentials v0.10 `ETModels.convert2et_full` returns them in the order
`(ETOneBody, ETPairModel, ETACE)`, each wrapped in a `WrappedSiteCalculator`.
"""
function cantor_substacks(fx)
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
        update cantor_substacks in export/test/fixtures/cantor_fixture.jl."""
    return (calcs[i1], calcs[ipr], calcs[iace])
end

"""
    cantor_mb_stack(fx) -> StackedCalculator

The `E0 + many-body` stack (`ETOneBody` + `ETACE`, pair term dropped): the reference the
current `:polynomial` export actually reproduces, and the `b_mb` column of `ref_k.txt`.
Components are located with [`cantor_substacks`](@ref), never by index.
"""
function cantor_mb_stack(fx)
    onebody_calc, _, ace_calc = cantor_substacks(fx)
    return ETM.StackedCalculator((onebody_calc, ace_calc))
end

"""
    cantor_spline_stack(fx; Nspl) -> StackedCalculator

`E0 + splinified ETACE`, built exactly as `chain_cantor.jl:129-136` builds it (this is the
reference for `:hermite_spline` exports; `Nspl = 50` -> `c50`, `Nspl = 200` -> `c200`).
Components are located with [`cantor_substacks`](@ref), never by index.
"""
function cantor_spline_stack(fx; Nspl::Integer)
    onebody_calc, _, ace_calc = cantor_substacks(fx)
    m = ETM.splinify(ace_calc.model, ace_calc.ps, ace_calc.st; Nspl = Nspl)
    p, s = LuxCore.setup(MersenneTwister(1), m)
    p.readout.W .= ace_calc.ps.readout.W
    return ETM.StackedCalculator((onebody_calc, ETM.ETACEPotential(m, p, s, fx.rcut)))
end
