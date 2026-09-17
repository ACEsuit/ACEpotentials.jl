# Evaluate an exported model file (a generated .jl source) on a set of configurations and
# compare energy / forces / virial against a Julia AtomsCalculators calculator.
#
# Public API (relied on by later tasks -- do not rename):
#   site_sets(sys, rcut)                            -> Vector{Tuple{Rs, Zs, Z0, js}}
#   site_sets(systems::AbstractVector, rcut)        -> Vector{Vector{...}}  (one entry per system)
#   exported_efv(ex, sys, rcut)                     -> (E::Float64, F::Vector{SVector{3}}, V::SMatrix{3,3})
#   check_export_report(model_file, calc, held, rcut; label) -> (maxdE, maxdF, maxdV)   [no assert]
#   check_export(model_file, calc, held, rcut; tol = 1e-12, label) -> (maxdE, maxdF, maxdV) [asserts]
#
# The two differ ONLY in who raises: check_export is check_export_report plus one @assert over
# all three maxima.  In-suite call sites use check_export_report + three explicit @test lines
# (same three quantities, same tol, but a breach reports as a test FAILURE naming the quantity
# instead of as an error).  Either way, gating all three is the CALLER's responsibility.
#
# METRICS -- READ THIS BEFORE CHOOSING A `tol`.  The three maxima are NOT normalised the same
# way, so one `tol` value means three different things:
#   maxdE = max |E - Eref| / natoms          [eV / atom]   PER ATOM  (energy is extensive)
#   maxdF = max_i |F_i - Fref_i|             [eV / Å]      ABSOLUTE  (a force is intensive)
#   maxdV = max |V - Vref|_inf / natoms      [eV / atom]   PER ATOM  (virial is extensive)
# The per-atom virial DELIBERATELY departs from a raw `maximum(abs.(V .- Vref))`: the Cantor
# virial is ~1.8e3 eV summed over ~4300 edges, so an absolute 1e-12 eV gate on it is ~4 ulp of
# double precision -- below the resolution of the arithmetic, hence no gate at all.  The full
# rationale, the measured headroom (1.23e-13 per atom vs tol 1e-12; 8e-16..3.6e-15 relative to
# |V|) and the residual risk this leaves are in the `check_export_report` docstring below.
# Both virial figures are printed on every call, each labelled with whether it is gated.
#
#
# ============================================================================================
# THE 1e-12 ABSOLUTE FORCE GATE IS BELOW DOUBLE-PRECISION RESOLUTION ON ILL-CONDITIONED MODELS
# ============================================================================================
#
# READ THIS BEFORE CONCLUDING THAT A CHANGE WHICH JUST MISSES THIS GATE IS WRONG.
#
# Measured (Task 7, and re-derived independently by review at 256-bit): on the TiAl order-4
# benchmark model, evaluated against a BigFloat evaluation of the SAME expressions,
#
#   * the intermediate `dA = dE/dA` carries ~1.2e-12 of ABSOLUTE error in Float64 whatever the
#     association -- its cancellation condition number kappa = sum|contributions| / |dA| is
#     18.8 (Ti) and 37.7 (Al) on |dA| of 317 and 152, so the floor kappa*eps*|dA| is
#     1.154e-12 / 1.228e-12;
#   * the SHIPPED generator's own error against exact arithmetic is 1.307e-12 (Ti) --
#     LARGER THAN THE 1e-12 GATE IT PASSES.
#
# It passes because it shares EquivariantTensors' product association and the two identical
# roundings cancel.  So on this model the gate measures AGREEMENT WITH THE REFERENCE'S
# ASSOCIATION, not accuracy: any correctly re-associated evaluation disagrees with the
# reference by the SUM of two independent ~1.2e-12 errors and reads ~1.7e-12 here.
#
# The gate is deliberately UNCHANGED (plan ruling, Task 7 fix round 1): the shipped default
# passes it, and the only thing it rejects is an evaluation route that was not adopted for
# independent (performance) reasons.  This note exists so the next person who hits it does not
# have to re-derive the analysis.  The measurement, the per-species table and the regenerating
# script are in `.superpowers/sdd/lammps_export_parity_plan/task-7-report.md` section 4b,
# `export/bench/README.md` ("The TiAl :dag libraries were NOT built and NOT timed") and
# `export/bench/diag_dA_conditioning.jl`.
#
# By contrast the Cantor model's kappa is 2.5-7.1 on |dA| of 10-30, so its floor is ~1e-14 and
# this gate has three orders of magnitude of headroom there.  The defect is a property of the
# MODEL, not of the metric everywhere.
# The neighbour-set construction follows verify_cantor/chain_cantor.jl:117-123 / 179-194, which
# is the code that produced verify_cantor/ref_{1..10}.txt -- that script, not any sketch, is the
# reference for units (plain Å, no Unitful), ordering (NeighbourLists.neigs order) and for the
# fact that the exported site energy already contains E0 of the central atom.

include(joinpath(@__DIR__, "fixtures", "cantor_fixture.jl"))

using AtomsCalculators: potential_energy, forces, virial
using LinearAlgebra: norm

"""
    site_sets(sys, rcut) -> Vector{Tuple{Vector{SVector{3,Float64}}, Vector{Int}, Int, Vector{Int}}}

Full neighbour set `(Rs, Zs, Z0, js)` for every site of `sys` within `rcut` (Å).
`Rs` are relative positions `r_j - r_i` in plain Float64 Å (no Unitful units -- that is what the
exported `site_energy_*` functions take), `Zs` the neighbour atomic numbers, `Z0` the centre's
atomic number and `js` the neighbour indices into `sys` (periodic images map back to the index
of the atom in the cell, which is what makes the force accumulation in `exported_efv` correct).
"""
function site_sets(sys, rcut)
    X = [SVector{3,Float64}(ustrip.(u"Å", p)) for p in position(sys, :)]
    C = Matrix(ustrip.(u"Å", hcat(cell_vectors(sys)...)'))
    nl = NL.PairList(X, rcut, C, (true, true, true))
    Zs_all = atomic_number(sys, :)
    out = Vector{Tuple{Vector{SVector{3,Float64}}, Vector{Int}, Int, Vector{Int}}}(undef, length(X))
    for i in 1:length(X)
        js, Rs = NL.neigs(nl, i)
        out[i] = ([SVector{3,Float64}(R) for R in Rs],
                  [Int(Zs_all[j]) for j in js], Int(Zs_all[i]), collect(js))
    end
    return out
end

"Per-system neighbour sets for a collection of systems."
site_sets(systems::AbstractVector, rcut) = [site_sets(sys, rcut) for sys in systems]

"""
    exported_efv(ex, sys, rcut) -> (E, F, V)

Total energy (eV), forces (eV/Å, one `SVector{3,Float64}` per atom) and virial (eV, a
`SMatrix{3,3,Float64,9}`) of `sys` obtained by calling `ex.site_energy_forces_virial` -- where
`ex` is a `Module` into which an exported model file has been `include`d -- on every site.

The exported site energy already includes E0 of the central atom, so the totals are directly
comparable to a stack that contains the `ETOneBody` term.
"""
function exported_efv(ex, sys, rcut)
    N = length(sys)
    E = 0.0
    F = zeros(SVector{3,Float64}, N)
    V = zero(SMatrix{3,3,Float64,9})
    for (i, (Rs, Zs, Z0, js)) in enumerate(site_sets(sys, rcut))
        Ei, Fi, Vi = ex.site_energy_forces_virial(Rs, Zs, Z0)
        E += Ei
        V += Vi
        for (k, j) in enumerate(js)
            F[j] += Fi[k]
            F[i] -= Fi[k]
        end
    end
    return E, F, V
end

"""
    load_exported(model_file) -> Module

`include` an exported model source into a fresh anonymous `Module`.
"""
function load_exported(model_file)
    ex = Module(Symbol("Exported_", hash(abspath(model_file))))
    Base.include(ex, abspath(model_file))
    return ex
end

"""
    check_export_report(model_file, calc, held, rcut; label = model_file) -> (maxdE, maxdF, maxdV)

Non-asserting variant of [`check_export`](@ref): evaluates the exported model on every
configuration of `held`, compares to the Julia calculator `calc`, prints and returns three
maxima taken over all configurations.

# What the three returned numbers are

| return value | definition | unit | normalisation |
|---|---|---|---|
| `maxdE` | `max \\|E - Eref\\| / natoms` | eV / atom | **per atom** |
| `maxdF` | `max_i \\|\\|F_i - Fref_i\\|\\|` | eV / Å | **absolute** |
| `maxdV` | `max \\|V - Vref\\|_inf / natoms` | eV / atom | **per atom** |

`maxdE` and `maxdV` are per atom because energy and virial are **extensive**: for the Cantor
reference system the total virial is ~1.8e3 eV accumulated over ~4300 edges, so an absolute
1e-12 eV gate on it sits at ~4 ulp of double precision — below the resolution of the arithmetic
and therefore not a gate at all; per atom (~38 eV) is the direct analogue of the per-atom energy
the plan already uses. `maxdF` is absolute because a force is **intensive**: it does not grow
with system size, so dividing by `natoms` would make the gate weaker on larger cells.

# Headroom — the per-atom gate is real, not vacuous

The exported `:polynomial` model against the `E0 + many-body` ET stack deviates by
`1.23e-13` eV/atom in the virial, against a `tol = 1e-12` — a factor ~8 of margin. The same
deviation is `8e-16 .. 3.6e-15` **relative** to `\\|V\\|_inf`, i.e. 4-16 ulp, which is what pure
summation roundoff looks like. A dropped pair term, by contrast, reads `3.8e+01` eV/atom.

# Residual risk, and WHOSE job it is to gate the virial

An absolute virial error smaller than `tol * natoms` eV (up to ~4.8e-11 eV for the 48-atom
configurations here) passes any gate built on `maxdV`. Nothing re-checks it afterwards, so
whichever of the two entry points a caller uses, **the caller is responsible for gating all
three returned quantities** — a call site that tests only `maxdF` leaves the virial
unconstrained entirely.

The two entry points differ only in who raises:

| | gate mechanism | reports as | use when |
|---|---|---|---|
| `check_export_report` + `@test x <= tol` | the caller's `@test` | a test **failure**, naming the quantity | inside a `@testset` |
| `check_export` | this file's internal `@assert` | a test **error** | outside a testset, or in a script |

Both are equally strict at the same `tol`. As of Task 2 the four in-suite call sites use
`check_export_report` + three explicit `@test` lines, so that the gate is visible in the test
summary and a breach names which of E, F or V moved; `check_export` remains for callers that
want the assertion, and Tasks 5-7 are briefed to use it.

Every call prints all four figures with unambiguous labels, each tagged `[per atom]` or
`[absolute]`, and the raw `max|dV|` is tagged `[absolute; reported only, never gated]` so it
cannot be mistaken for a gated quantity.

Use `check_export_report` when the deviation *is* the measurement (e.g. the Hermite-spline
error against the fitted model, which must never be asserted against a tolerance at all).
"""
function check_export_report(model_file, calc, held, rcut; label = model_file)
    ex = load_exported(model_file)
    maxdE = maxdF = maxdV = maxdVabs = 0.0
    for sys in held
        N = length(sys)
        E, F, V = Base.invokelatest(exported_efv, ex, sys, rcut)
        Eref = ustrip(u"eV", potential_energy(sys, calc))
        Fref = [SVector{3,Float64}(ustrip.(u"eV/Å", f)) for f in forces(sys, calc)]
        Vref = SMatrix{3,3,Float64,9}(ustrip.(u"eV", virial(sys, calc)))
        dV = maximum(abs.(V .- Vref))
        maxdE = max(maxdE, abs(E - Eref) / N)
        maxdF = max(maxdF, maximum(norm.(F .- Fref)))
        maxdV = max(maxdV, dV / N)
        maxdVabs = max(maxdVabs, dV)
    end
    println("$label:\n    max|dE|/atom = $maxdE eV/atom   [per atom; returned as maxdE]" *
            "\n    max|dF|      = $maxdF eV/Å   [absolute; returned as maxdF]" *
            "\n    max|dV|/atom = $maxdV eV/atom   [per atom; returned as maxdV]" *
            "\n    max|dV|      = $maxdVabs eV   [absolute; reported only, never gated]")
    flush(stdout)
    return (maxdE, maxdF, maxdV)
end

"""
    check_export(model_file, calc, held, rcut; tol = 1e-12, label = model_file)

As [`check_export_report`](@ref), but asserts that all three maxima are `<= tol`.
Returns `(maxdE, maxdF, maxdV)`.  **Never loosen `tol` to make a call pass.**

The three numbers are *not* normalised the same way, and `tol` therefore means three different
things — read the table in [`check_export_report`](@ref) before choosing one:

| gated quantity | normalisation | `tol = 1e-12` means |
|---|---|---|
| `maxdE` | per atom | 1e-12 eV/atom |
| `maxdF` | absolute | 1e-12 eV/Å |
| `maxdV` | per atom | 1e-12 eV/atom, i.e. up to `1e-12 * natoms` eV in total |

That last row is deliberate (energy and virial are extensive; see
[`check_export_report`](@ref) for why an absolute virial gate at this magnitude is below
double-precision resolution) and it is the residual risk of this harness: a total virial error
under `tol * natoms` eV is not caught.

This function is [`check_export_report`](@ref) plus one `@assert` over all three maxima. A
caller inside a `@testset` will usually prefer `check_export_report` and three explicit
`@test` lines, which gate the same three quantities at the same `tol` but report a breach as a
test failure naming the quantity rather than as an error; the in-suite call sites do that as
of Task 2. Either way the virial is only constrained if the caller constrains it.
"""
function check_export(model_file, calc, held, rcut; tol = 1e-12, label = model_file)
    maxdE, maxdF, maxdV = check_export_report(model_file, calc, held, rcut; label = label)
    println("    (tol = $tol, applied to max|dE|/atom, max|dF| absolute, max|dV|/atom)")
    @assert maxdE <= tol && maxdF <= tol && maxdV <= tol "$label exceeds tol=$tol: dE=$maxdE dF=$maxdF dV=$maxdV"
    return (maxdE, maxdF, maxdV)
end
