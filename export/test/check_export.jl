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
# METRICS.  Energy and virial are extensive, forces are not, so the three maxima are
#   maxdE = max |E - Eref| / natoms          [eV / atom]
#   maxdF = max_i |F_i - Fref_i|             [eV / Å]
#   maxdV = max |V - Vref|_inf / natoms      [eV / atom]
# The per-atom normalisation of the virial is NOT a loosened tolerance: the Cantor virial is
# ~1.7e3 eV summed over ~4300 edges, so pure double-precision summation roundoff is ~3e-12
# absolute (= 3e-15 relative, ~10 ulp) even when every component matches bit-for-bit in intent.
# Per atom that is <= 1.3e-13, comfortably inside 1e-12.  The absolute figure is printed
# alongside so nothing is hidden.
#
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
configuration of `held`, compares to `calc`, prints and returns the maxima

* `maxdE` -- max |ΔE| **per atom** (eV/atom)
* `maxdF` -- max over atoms and configs of `norm(F - Fref)` (eV/Å)
* `maxdV` -- max over components of `|V - Vref|` **per atom** (eV/atom); see the METRICS note
  at the top of this file.  The un-normalised maximum is printed as `max|dV|abs`.

Use this when the deviation is the measurement (e.g. Hermite-vs-fitted error); use
`check_export` when the deviation must be inside a tolerance.
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
    println("$label: max|dE|/atom = $maxdE  max|dF| = $maxdF  max|dV|/atom = $maxdV  (max|dV|abs = $maxdVabs)")
    flush(stdout)
    return (maxdE, maxdF, maxdV)
end

"""
    check_export(model_file, calc, held, rcut; tol = 1e-12, label = model_file)

As [`check_export_report`](@ref), but asserts that all three maxima are `<= tol`.
Returns `(maxdE, maxdF, maxdV)`.  Never loosen `tol` to make a call pass.
"""
function check_export(model_file, calc, held, rcut; tol = 1e-12, label = model_file)
    maxdE, maxdF, maxdV = check_export_report(model_file, calc, held, rcut; label = label)
    println("    (tol $tol)")
    @assert maxdE <= tol && maxdF <= tol && maxdV <= tol "$label exceeds tol=$tol: dE=$maxdE dF=$maxdF dV=$maxdV"
    return (maxdE, maxdF, maxdV)
end
