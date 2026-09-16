#=
Why the TiAl :polynomial VIRIAL moves by more than 1e-13 relative under Task 6's kernel
restructuring, and why that is re-association roundoff rather than a defect.

This is a DIAGNOSTIC, not a gate.  It is kept because the claim it supports -- "the one
parity quantity that exceeds 1e-13 does so because of the virial's own conditioning, not
because the new kernel computes a different thing" -- has to be checkable, not asserted.

    cd export/test && julia --project=.. diag_virial_conditioning.jl

It measures three things on the TiAl held-out configurations:

 1. kappa = (sum over every site and neighbour of |R_j (x) f_j|_inf) / |V|_inf.
    The virial is a SUM OF SIGNED RANK-1 TERMS that very nearly cancels.  A relative error
    `eps_F` in the forces therefore shows up in V as roughly `eps_F * kappa` relative to |V|.
    kappa is a property of the MODEL AND THE CELL -- it does not depend on which generator
    produced the forces -- so it is the honest scale against which to read the observed dV.

 2. |V_new - V_calc| and |V_old - V_calc| against the ETACE/Stacked Julia calculator, which
    computes the virial by a completely different route (EquivariantTensors' own pullbacks).
    If the restructuring had introduced a defect, the distance to this INDEPENDENT reference
    would grow.  If it is roundoff, the two distances are comparable and both stay far inside
    the 1e-12 export gate.

 3. The same for the forces, so that the virial figure can be read against the force figure.
=#

using Printf, LinearAlgebra, StaticArrays

include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(@__DIR__, "fixtures", "tial_fixture.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

const PB = joinpath(@__DIR__, "build", "parity")
const F_OLD = joinpath(PB, "tial_poly_ref.jl")
const F_NEW = joinpath(PB, "tial_poly_new.jl")

for f in (F_OLD, F_NEW)
    isfile(f) || error("""
        $f does not exist.  Run the parity gate first, which writes both generators' output:
            cd export/test && EXPORT_REF_SHA=<prev task sha> julia --project=.. \\
                -e 'include("test_generator_parity.jl")'""")
end

fx = load_tial_fixture()
ex_old = Base.invokelatest(load_exported, F_OLD)
ex_new = Base.invokelatest(load_exported, F_NEW)

using AtomsCalculators: virial, forces
using Unitful

println("\n", "="^100)
println("TiAl :polynomial -- virial conditioning and distance to an independent reference")
println("="^100)

function sweep(fx, ex_old, ex_new)
kmax = 0.0
dV_new = dV_old = dV_gen = 0.0
dF_new = dF_old = dF_gen = 0.0

for (n, sys) in enumerate(fx.held)
    E_o, F_o, V_o = Base.invokelatest(exported_efv, ex_old, sys, fx.rcut)
    E_n, F_n, V_n = Base.invokelatest(exported_efv, ex_new, sys, fx.rcut)

    Vref = ustrip.(u"eV", virial(sys, fx.stacked))
    Fref = [ustrip.(u"eV/Å", f) for f in forces(sys, fx.stacked)]

    # kappa: the non-cancelling magnitude of the rank-1 terms the virial is a sum of,
    # against the size of the (heavily cancelled) total.  Uses the NEW forces; the OLD ones
    # give the same figure to three digits, which is the point -- kappa is a property of the
    # configuration, not of the generator.
    S = 0.0
    for (Rs, Zs, Z0, _js) in site_sets(sys, fx.rcut)
        Ei, Fi, _Vi = Base.invokelatest(ex_new.site_energy_forces_virial, Rs, Zs, Z0)
        for (R, f) in zip(Rs, Fi)
            S += maximum(abs.(R * f'))
        end
    end
    Vinf = maximum(abs.(Vref))
    kappa = S / Vinf
    kmax = max(kmax, kappa)

    dV_new = max(dV_new, maximum(abs.(V_n .- Vref)) / Vinf)
    dV_old = max(dV_old, maximum(abs.(V_o .- Vref)) / Vinf)
    dV_gen = max(dV_gen, maximum(abs.(V_n .- V_o)) / maximum(abs.(V_o)))

    fscale = max(maximum(norm.(Fref)), 1.0)
    dF_new = max(dF_new, maximum(norm.(F_n .- Fref)) / fscale)
    dF_old = max(dF_old, maximum(norm.(F_o .- Fref)) / fscale)
    dF_gen = max(dF_gen, maximum(norm.(F_n .- F_o)) / max(maximum(norm.(F_o)), 1.0))

    @printf("  config %2d: natoms %3d   |V|_inf %10.3f eV   Σ|R⊗f|_inf %12.1f eV   κ = %7.1f\n",
            n, length(sys), Vinf, S, kappa)
end
return (; kmax, dF_gen, dV_gen, dF_old, dV_old, dF_new, dV_new)
end

r = sweep(fx, ex_old, ex_new)
kmax, dF_gen, dV_gen = r.kmax, r.dF_gen, r.dV_gen
dF_old, dV_old, dF_new, dV_new = r.dF_old, r.dV_old, r.dF_new, r.dV_new

println()
@printf("worst κ over the held-out set                         : %.1f\n", kmax)
@printf("generator-to-generator (the parity gate's quantity)   : dF %.3e   dV %.3e\n",
        dF_gen, dV_gen)
@printf("κ x the measured generator-to-generator force change  : %.3e   <- expected dV scale\n",
        kmax * dF_gen)
println()
@printf("distance to the INDEPENDENT ETACE calculator, relative to |V|_inf / |F|_max:\n")
@printf("    OLD generator: dF %.3e   dV %.3e\n", dF_old, dV_old)
@printf("    NEW generator: dF %.3e   dV %.3e\n", dF_new, dV_new)
println()
println(dV_new <= dV_old ?
        "The NEW generator is AT LEAST AS CLOSE to the independent reference as the OLD one." :
        "The NEW generator is FURTHER from the independent reference than the OLD one -- " *
        "this is evidence of a defect, not of roundoff.  Investigate.")
println("="^100)
