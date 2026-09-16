# fit_tial_order4.jl -- fit the LARGE reference model used by the parity benchmarks.
#
# Ti-Al, correlation order 4, rcut 5.5 Å, BLR + repulsion restraint.  The point of this model
# is to be *big*: the Cantor reference has 5 species but only order 3 / 1078 many-body basis
# functions, which is too small to show the cost structure the performance tasks are chasing.
# `max_level` is chosen so that the many-body basis lands in 1500-2500 functions (see the
# `--probe` output recorded in export/bench/README.md).
#
# Run (detached; ~10-60 min):
#   cd <repo> && mkdir -p bench_parity
#   nohup julia --project=export export/bench/fit_tial_order4.jl > bench_parity/fit_tial.log 2>&1 &
#
# Options (argv):
#   --probe       print length(model.tensor) for max_level = 8..13 and exit (no fit)
#
# Output:
#   bench_parity/tial_o4_params.jld2   with `ps`, `st`, `E0s`, `train_idx`, `held_idx`,
#                                      `check_idx`, `hypers`
# The fixture export/test/fixtures/tial_fixture.jl rebuilds the model from that file; its
# model-construction block is a verbatim copy of TIAL_MODEL below and must stay identical.

const EXPORT_PROJECT = normpath(joinpath(@__DIR__, ".."))     # <repo>/export
using Pkg; Pkg.activate(EXPORT_PROJECT)
using Distributed
const NPROC = parse(Int, get(ENV, "ACE_FIT_NPROC", "8"))
const PROBE_ONLY = "--probe" in ARGS
if !PROBE_ONLY && NPROC > 1
    # ACEfit assembles the least-squares system over Distributed workers (as chain_cantor.jl does)
    addprocs(NPROC; exeflags = "--project=$EXPORT_PROJECT")
    @everywhere using ACEpotentials
end

using ACEpotentials, ACEfit, AtomsBase, StaticArrays, LinearAlgebra, Random
using Lux, JLD2, Printf
using AtomsBase: atomic_number

const M = ACEpotentials.Models
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "bench_parity")
mkpath(OUT)

# ---------------------------------------------------------------- hyper-parameters
# Everything the fixture needs in order to rebuild this exact model.
const TIAL_ELEMENTS  = (:Ti, :Al)
const TIAL_ORDER     = 4
const TIAL_RCUT      = 5.5
const TIAL_MAX_LEVEL = 11        # -> 2369 many-body basis functions (1500 <= n <= 2500)
const TIAL_PAIR_MAXN = 20
const TIAL_WL        = 1.5       # ace1_model's default level weighting
# The tutorial's one-body reference energies (docs/src/tutorials/basic_julia_workflow.jl:36).
const TIAL_E0S = Dict(:Ti => -1586.0195, :Al => -105.5954)

"""
    tial_model(E0s = TIAL_E0S) -> ACEpotentials.Models.ACEModel

THE model-construction block.  `export/test/fixtures/tial_fixture.jl` carries a verbatim copy;
the saved `ps` only fits this exact spec, so any edit here must be mirrored there (otherwise it
surfaces as a Lux parameter-shape error, or -- worse -- a silently wrong energy).

`E0s` is not merely an initialisation: `ETModels.convert2et_full` reads `model.Vref.E0` to build
the `ETOneBody` term of the exported stack.
"""
function tial_model(E0s = TIAL_E0S; max_level = TIAL_MAX_LEVEL)
    NZ = length(TIAL_ELEMENTS)
    r0 = M._default_rin0cuts(TIAL_ELEMENTS)     # per-pair bond lengths; rcut overridden to 5.5
    rin0cuts = SMatrix{NZ, NZ}([(rin = 0.0, r0 = r0[i, j].r0, rcut = TIAL_RCUT)
                                for i in 1:NZ, j in 1:NZ])
    level = M.TotalDegree(1.0 * NZ, 1 / TIAL_WL)
    return M.ace_model(; elements = TIAL_ELEMENTS, order = TIAL_ORDER, Ytype = :solid,
                         level = level, max_level = max_level,
                         pair_maxn = TIAL_PAIR_MAXN, rin0cuts = rin0cuts,
                         init_WB = :zeros, init_Wpair = :onehot, init_Wradial = :onehot,
                         pair_learnable = true, E0s = E0s)
end

"""
    tial_split(data) -> (train_idx, held_idx, check_idx)

Deterministic split of the `TiAl_tutorial` dataset.  That dataset is **315 dimers + 14 bulk
configurations** (9 x 54 atoms, 5 x 128 atoms), so a naive `data[1:5:end]` train set is almost
entirely two-atom configurations and carries far too little information for a 4738-parameter
model.  Instead:

* `held_idx`  -- a genuine out-of-sample set: every 4th bulk configuration (4 of 14) plus every
                 16th dimer (20 of 315).  Reported by `compute_errors` as the held-out table.
* `train_idx` -- everything else (305 configurations, 10 of them bulk).
* `check_idx` -- the 10 configurations the *export accuracy gate* runs on: the 5 x 128-atom and
                 5 x 54-atom bulk cells.  These are geometric coverage for the 1e-12 export
                 comparison, NOT a statistical held-out set -- most of them are in `train_idx`.
                 (The gate compares exported code against the Julia calculator on the same
                 geometry, so in-sample/out-of-sample is irrelevant to it.)
"""
function tial_split(data)
    N = length.(data)
    bulk   = findall(>=(54), N)
    dimer  = findall(<(54), N)
    held   = sort(vcat(bulk[2:4:end], dimer[1:16:end]))
    train  = setdiff(1:length(data), held)
    big    = findall(==(128), N)          # 5
    medium = findall(==(54), N)[1:5]      # 5
    check  = sort(vcat(big, medium))
    return train, held, check
end

# ---------------------------------------------------------------- probe mode
if PROBE_ONLY
    println("many-body basis size vs max_level (order = $TIAL_ORDER, rcut = $TIAL_RCUT, ",
            "pair_maxn = $TIAL_PAIR_MAXN, level = TotalDegree($(1.0*length(TIAL_ELEMENTS)), 1/$TIAL_WL))")
    for ml in 8:13
        m = tial_model(; max_level = ml)
        @printf("  max_level = %2d :  many-body basis = %5d   rbasis (n,l) = %3d   pair = %3d   maxl = %d  maxn = %d\n",
                ml, length(m.tensor), length(m.rbasis), length(m.pairbasis),
                maximum(b.l for b in m.rbasis.spec), maximum(b.n for b in m.rbasis.spec))
        flush(stdout)
    end
    exit(0)
end

# ---------------------------------------------------------------- data
t_start = time()
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_idx, held_idx, check_idx = tial_split(data)
train = data[train_idx]; held = data[held_idx]
@printf("dataset: %d configs (%d atoms) -- %s\n", length(data), sum(length, data), meta)
@printf("train %d configs (%d atoms, %d bulk) | held-out %d configs (%d atoms) | export-check %d configs (%d atoms)\n",
        length(train), sum(length, train), count(>=(54), length.(train)),
        length(held), sum(length, held), length(check_idx), sum(length, data[check_idx]))
flush(stdout)

# ---------------------------------------------------------------- model
model = tial_model()
ps0, st0 = Lux.setup(MersenneTwister(1234), model)
pot = M.ACEPotential(model, ps0, st0)
@printf("basis: many-body %d per species, pair %d per species, rbasis %d (n,l), pairbasis %d, total linear params %d\n",
        size(ps0.WB, 1), size(ps0.Wpair, 1), length(model.rbasis), length(model.pairbasis),
        length(ps0.WB) + length(ps0.Wpair))
println("rbasis spec maxl = ", maximum(b.l for b in model.rbasis.spec),
        ", maxn = ", maximum(b.n for b in model.rbasis.spec))
flush(stdout)

# ---------------------------------------------------------------- fit
keys_ = (energy_key = "energy", force_key = "force", virial_key = "virial")
t0 = time()
BLAS.set_num_threads(NPROC)
acefit!(train, pot; solver = ACEfit.BLR(), repulsion_restraint = true, verbose = true, keys_...)
@printf("fit wall time %.0f s\n", time() - t0); flush(stdout)

println("--- train errors"); ACEpotentials.compute_errors(train, pot; keys_...)
println("--- held-out errors"); ACEpotentials.compute_errors(held, pot; keys_...)
println("--- export-check configs (in-sample; geometry coverage only)")
ACEpotentials.compute_errors(data[check_idx], pot; keys_...)
flush(stdout)

# ---------------------------------------------------------------- save (FIRST thing after the fit)
hypers = (; elements = TIAL_ELEMENTS, order = TIAL_ORDER, rcut = TIAL_RCUT,
            max_level = TIAL_MAX_LEVEL, pair_maxn = TIAL_PAIR_MAXN, wL = TIAL_WL)
JLD2.jldsave(joinpath(OUT, "tial_o4_params.jld2");
             ps = pot.ps, st = pot.st, E0s = TIAL_E0S,
             train_idx = train_idx, held_idx = held_idx, check_idx = check_idx, hypers = hypers)
println("saved ", joinpath(OUT, "tial_o4_params.jld2"))
flush(stdout)

# ---------------------------------------------------------------- physicality sanity
using AtomsCalculators: potential_energy, forces
using Unitful, Unitful: ustrip, @u_str
_F(f) = [SVector{3,Float64}(ustrip.(u"eV/Å", fi)) for fi in f]
for k in check_idx[[1, 6]]
    at = data[k]
    Fm = maximum(norm.(_F(forces(at, pot))))
    Fr = maximum(norm.([SVector{3,Float64}(f) for f in at[:, :force]]))
    @printf("config %d (%d atoms): max|F| fitted %.3f eV/Å   reference %.3f eV/Å\n",
            k, length(at), Fm, Fr)
end
@printf("TOTAL wall time %.0f s\nDONE fit_tial_order4.jl\n", time() - t_start)
