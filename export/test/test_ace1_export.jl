#=
Export of an `ace1_model`-derived ETACE stack -- NZ = 3, gated at 1e-12.

WHY THIS GROUP EXISTS.  `ace1_model` differs from the `ace_model` defaults every other group
exports in two ways, and the exporter used to handle neither:

  1. `Ytype = :spherical`: the angular basis is the real SPHERICAL harmonics
     Y_lm(r) = Z_lm(r / |r|), not the solid harmonics Z_lm(r).  The generator always
     emitted solid harmonics, and the export was then WRONG WITH NO ERROR (4.7e-2 eV/atom and
     1.38 eV/Å on a SiGe model).  The many-body-only testset below is the one that catches it:
     it has no pair term, so nothing else can fail first.
  2. the pair term uses the ACE1 envelope `ACE1_PolyEnvelope1sR`, with (rcut, r0) PER
     SPECIES PAIR, where the exporter only knew the species-independent `PolyEnvelope1sR`.

The model is the unsplinified "exact twin" of `ace1_model` (test/etmodels/ace1_exact_twin.jl):
`ace1_model` splinifies its radial bases, and a splinified model cannot be exported (see the
refusal in export_ace_model.jl).  The twin's basis is checked to be ace1_model's.

REFERENCES AND TOLERANCES (metric definitions are in check_export.jl): the exported module,
`include`d in-process, against the ET stack it was exported from AND against the ACEModel the
stack was converted from, both at 1e-12.  No tolerance in this file may ever be loosened.
=#

using Test
using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
using AtomsBase
using Unitful
import Polynomials4ML as P4ML
import EquivariantTensors as ET
import SpheriCart

include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "test", "etmodels", "ace1_exact_twin.jl"))

const A1_ELEMENTS = [:Si, :Ge, :C]
const A1_KW = (elements = A1_ELEMENTS, order = 3, totaldegree = 6,
               Eref = Dict(:Si => -1.25, :Ge => -2.5, :C => -3.75))

"The twin with random many-body and pair weights (the twin's init_WB is :zeros)."
function a1_model(; pair_basis = nothing)
    twin = ace1_exact_twin(; A1_KW...)
    if pair_basis !== nothing
        lvl, maxlvl = ACEpotentials.ACE1compat._get_degrees(
                          ACEpotentials.ACE1compat._clean_args(A1_KW))
        twin = M.ace_model(; elements = A1_ELEMENTS, order = A1_KW.order,
                             Ytype = :spherical, ZBL = false,
                             E0s = Dict(k => v * u"eV" for (k, v) in A1_KW.Eref),
                             rbasis = twin.rbasis, pair_basis = pair_basis,
                             rin0cuts = twin.rbasis.rin0cuts, level = lvl,
                             max_level = maxlvl, init_WB = :zeros)
    end
    ps, st = Lux.setup(MersenneTwister(11), twin)
    ps.WB .= 0.1 .* randn(MersenneTwister(12), size(ps.WB))
    ps.Wpair .= 0.1 .* randn(MersenneTwister(13), size(ps.Wpair))
    return twin, ps, st
end

"Rattled diamond supercells, species cycled Si/Ge/C so every ordered pair occurs."
function a1_configs(n)
    out = []
    for k in 1:n
        rng = MersenneTwister(200 + k)
        a = 5.43
        base = [SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.0), SVector(0.5, 0.0, 0.5),
                SVector(0.0, 0.5, 0.5), SVector(0.25, 0.25, 0.25), SVector(0.75, 0.75, 0.25),
                SVector(0.75, 0.25, 0.75), SVector(0.25, 0.75, 0.75)]
        pos = SVector{3,Float64}[]
        spec = Symbol[]
        idx = 0
        for ix in 0:1, iy in 0:1, b in base
            idx += 1
            push!(pos, a .* (SVector(ix, iy, 0) .+ b) + 0.12 * randn(rng, SVector{3,Float64}))
            push!(spec, A1_ELEMENTS[mod1(idx + k, 3)])
        end
        box = [SVector(2a, 0.0, 0.0), SVector(0.0, 2a, 0.0), SVector(0.0, 0.0, a)]
        push!(out, AtomsBase.periodic_system([e => x * u"Å" for (e, x) in zip(spec, pos)],
                                             box .* u"Å"))
    end
    return out
end

"The model's pair basis with r0 perturbed per ORDERED pair, so (i,j) and (j,i) differ."
function a1_asym_pairbasis(pb)
    NZ = length(pb._i2z)
    envs = [M.ACE1_PolyEnvelope1sR(pb.envelopes[i, j].rcut,
                                   pb.envelopes[i, j].r0 * (1 + 0.07 * i - 0.04 * j),
                                   pb.envelopes[i, j].p) for i in 1:NZ, j in 1:NZ]
    return M.LearnableRnlrzzBasis(pb._i2z, pb.polys, pb.transforms, SMatrix{NZ,NZ}(envs),
                                  pb.rin0cuts, pb.spec; Winit = :onehot)
end

function a1_refusal(calc, f)
    isfile(f) && rm(f)
    err = try
        Base.invokelatest(export_ace_model, calc, f)
        nothing
    catch e
        e
    end
    return (; err, msg = err === nothing ? "" : sprint(showerror, err), f)
end

@testset "ace1_model-derived ETACE export" verbose = true begin

    model, ps, st = a1_model()
    # convert2et_full's cutoff.  The stack itself is built INSIDE the testsets that use it,
    # so that a conversion failure cannot mask the many-body-only gate, which needs none.
    rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)
    a1_stack() = ETM.convert2et_full(model, ps, st)
    held = a1_configs(3)
    build = mkpath(joinpath(@__DIR__, "build"))
    acepot = M.ACEPotential(model, ps, st)

    @testset "the stack is ace1_model's" begin
        @test ace1_twin_matches(model, ace1_model(; A1_KW...).model)
        stacked = a1_stack()
        @test stacked.calcs[end].rcut == rcut
        @test [string(nameof(typeof(c.model))) for c in stacked.calcs] ==
              ["ETOneBody", "ETPairModel", "ETACE"]
        ybasis = stacked.calcs[end].model.yembed.layer.basis
        @test ybasis isa P4ML.RealSCWrapper
        @test ybasis.scbasis isa SpheriCart.SphericalHarmonics
        @test all(e isa M.ACE1_PolyEnvelope1sR for e in model.pairbasis.envelopes)
        @test length(unique(e.r0 for e in model.pairbasis.envelopes)) == 6
    end

    @testset "many-body only (spherical harmonics), vs ETACEPotential, 1e-12" begin
        # No pair term and no E0: the ONLY thing this export can get wrong that the
        # existing groups would not already catch is the angular basis.
        et = ETM.convert2et(model)
        et_ps, et_st = LuxCore.setup(MersenneTwister(1), et)
        ETM.copy_ace_params!(et_ps, ps, model)
        calc = ETM.ETACEPotential(et, et_ps, et_st, rcut)
        f = joinpath(build, "ace1_mb_only.jl")
        Base.invokelatest(export_ace_model, calc, f)
        @test isfile(f)
        dE, dF, dV = Base.invokelatest(check_export_report, f, calc, held, rcut;
                                       label = "ace1 many-body only vs ETACEPotential")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12

        # and the emitted angular basis itself, against the model's, value and gradient
        ex = Base.invokelatest(load_exported, f)
        ybasis = et.yembed.layer.basis
        rng = MersenneTwister(3)
        errY = errdY = 0.0
        for _ in 1:20
            R = (1.0 + 4.0 * rand(rng)) * normalize(randn(rng, SVector{3,Float64}))
            Y, dY = P4ML.evaluate_ed(ybasis, R)
            Ye, dYe = Base.invokelatest(ex.eval_ylm_ed, R)
            errY = max(errY, maximum(abs.(Ye .- Y)))
            errdY = max(errdY, maximum(norm.(dYe .- dY)))
            @test Base.invokelatest(ex.eval_ylm, R) ≈ Ye rtol = 0 atol = 1e-14
        end
        @info "ace1 harmonics: max|dY| = $errY, max|d∇Y| = $errdY"
        @test errY <= 1e-13
        @test errdY <= 1e-13
    end

    f_full = joinpath(build, "ace1_full.jl")
    isfile(f_full) && rm(f_full)

    @testset "E0 + ACE1 pair + spherical ETACE vs the ET stack, 1e-12" begin
        stacked = a1_stack()
        Base.invokelatest(export_ace_model, stacked, f_full)
        @test isfile(f_full)
        dE, dF, dV = Base.invokelatest(check_export_report, f_full, stacked, held, rcut;
                                       label = "ace1 full stack vs ET stack")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12
    end

    @testset "E0 + ACE1 pair + spherical ETACE vs the ACEModel, 1e-12" begin
        isfile(f_full) || Base.invokelatest(export_ace_model, a1_stack(), f_full)
        dE, dF, dV = Base.invokelatest(check_export_report, f_full, acepot, held, rcut;
                                       label = "ace1 full stack vs ACEModel")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12
    end

    @testset "asymmetric ACE1 pair envelope (ordered pairs), 1e-12" begin
        model_a, ps_a, st_a = a1_model(; pair_basis = a1_asym_pairbasis(model.pairbasis))
        envs = model_a.pairbasis.envelopes
        @test any(envs[i, j].r0 != envs[j, i].r0 for i in 1:3, j in 1:3)
        stacked_a = ETM.convert2et_full(model_a, ps_a, st_a)
        f = joinpath(build, "ace1_asym_pair.jl")
        Base.invokelatest(export_ace_model, stacked_a, f)
        dE, dF, dV = Base.invokelatest(check_export_report, f, stacked_a, held, rcut;
                                       label = "ace1 asym pair envelope vs ET stack")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12
        dE, dF, dV = Base.invokelatest(check_export_report, f, M.ACEPotential(model_a, ps_a, st_a),
                                       held, rcut; label = "ace1 asym pair envelope vs ACEModel")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12
    end

    @testset "an angular basis the generator cannot express is refused" begin
        # many-body only, so this runs even where the pair conversion does not
        et_m = ETM.convert2et(model)
        et_ps, et_st = LuxCore.setup(MersenneTwister(1), et_m)
        ETM.copy_ace_params!(et_ps, ps, model)
        et = ETM.ETACEPotential(et_m, et_ps, et_st, rcut)
        function with_ybasis(yb)
            m = et.model
            yembed = ET.EdgeEmbed(ET.EmbedDP(ET.NTtransformST((x, st) -> x.𝐫, NamedTuple()), yb))
            m2 = ETM.ETACE(m.rembed, yembed, m.basis, m.readout)
            return ETM.ETACEPotential(m2, et.ps, et.st, et.rcut)
        end
        L = P4ML.maxl(et.model.yembed.layer.basis)
        for (tag, yb, needle) in (
                ("racah", P4ML.real_sphericalharmonics(L; normalisation = :racah), "racah"),
                ("racah_solid", P4ML.real_solidharmonics(L; normalisation = :racah), "racah"),
                ("complex", P4ML.complex_sphericalharmonics(L), "ComplexSCWrapper"))
            r = a1_refusal(with_ybasis(yb), joinpath(build, "ace1_refused_$tag.jl"))
            @test r.err isa ErrorException
            @test occursin(needle, r.msg)
            @test isfile(r.f) == false
        end
    end

    @testset "a pair envelope the generator cannot express is refused" begin
        stacked = a1_stack()
        pc = stacked.calcs[2]
        branch = pc.model.rembed.layer
        env = ET.dp_transform((x, st) -> st.c * norm(x.𝐫), (c = 1.0,))
        pm2 = ETM.ETPairModel(ET.EdgeEmbed(ETM.EnvRBranchL(env, branch.rbasis)), pc.model.readout)
        ps2, st2 = LuxCore.setup(MersenneTwister(1), pm2)
        ps2.rembed.rbasis.post.W .= pc.ps.rembed.rbasis.post.W
        ps2.readout.W .= pc.ps.readout.W
        pcalc2 = ETM.WrappedSiteCalculator(pm2, ps2, st2, pc.rcut)
        s2 = ETM.StackedCalculator((stacked.calcs[1], pcalc2, stacked.calcs[end]))
        r = a1_refusal(s2, joinpath(build, "ace1_refused_pairenv.jl"))
        @test r.err isa ErrorException
        @test occursin("pair envelope", r.msg)
        @test isfile(r.f) == false
    end
end
