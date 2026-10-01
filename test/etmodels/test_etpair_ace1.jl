# ETModels conversion of an `ace1_model`-style pair term.
#
# `ace1_model` builds its pair basis with the ACE1 envelope `ACE1_PolyEnvelope1sR`, whose
# (rcut, r0) are set PER SPECIES PAIR (r0 = the pair's bond length):
#
#     env(r) = s^-p - sc^-p + p sc^(-p-1) (s - sc),   s = r / r0,  sc = rcut / r0,
#     env(r) = 0 for r > rcut.
#
# `ETModels.convertpair` must reproduce it per ORDERED pair (centre species first, the
# convention `LearnableRnlrzzBasis` indexes `envelopes[iz, jz]` with).  It used to accept
# only the species-independent `PolyEnvelope1sR`, so converting an ace1_model failed.
#
# The model is the unsplinified "exact twin" of ace1_model (ace1_exact_twin.jl): the
# conversion needs the polynomial basis the splines were tabulated from.

using ACEpotentials, StaticArrays, Lux, AtomsBase, Unitful,
      AtomsCalculators, Random, LuxCore, Test, LinearAlgebra

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

include(joinpath(@__DIR__, "ace1_exact_twin.jl"))

const ACE1P_ELEMENTS = [:Si, :Ge, :C]
const ACE1P_KW = (elements = ACE1P_ELEMENTS, order = 2, totaldegree = 6)

"Rattled 3-species diamond supercell (2x2x1 cubic cells); every ordered species pair occurs."
function ace1p_struct(seed)
   rng = MersenneTwister(seed)
   a = 5.43
   base = [SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.0), SVector(0.5, 0.0, 0.5),
           SVector(0.0, 0.5, 0.5), SVector(0.25, 0.25, 0.25), SVector(0.75, 0.75, 0.25),
           SVector(0.75, 0.25, 0.75), SVector(0.25, 0.75, 0.75)]
   X = [a * (SVector(ix, iy, 0) + b) + 0.15 * randn(rng, SVector{3, Float64})
        for ix in 0:1 for iy in 0:1 for b in base]
   Z = [ACE1P_ELEMENTS[mod1(i + seed, 3)] for i in 1:length(X)]
   box = [SVector(2a, 0.0, 0.0), SVector(0.0, 2a, 0.0), SVector(0.0, 0.0, a)]
   return periodic_system([z => x * u"Å" for (z, x) in zip(Z, X)], box .* u"Å")
end

"Pair-only reference: the ACEModel with every many-body weight zero."
function ace1p_pair_only(model)
   ps, st = Lux.setup(MersenneTwister(11), model)
   ps.WB .= 0
   ps.Wpair .= 0.1 .* randn(MersenneTwister(12), size(ps.Wpair))
   return ps, st
end

function ace1p_compare(model, ps, st; ntest = 3)
   et_pair = ETM.convertpair(model)
   et_ps, et_st = Lux.setup(MersenneTwister(1), et_pair)
   ETM.copy_pair_params!(et_ps, ps, model)
   rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)
   calc_et = ETM.ETPairPotential(et_pair, et_ps, et_st, rcut)
   calc_ace = M.ACEPotential(model, ps, st)
   errs = (dE = 0.0, dF = 0.0, dV = 0.0)
   for k in 1:ntest
      sys = ace1p_struct(k)
      nat = length(sys)
      E1 = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_ace))
      E2 = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_et))
      F1 = ustrip.(u"eV/Å", reduce(hcat, AtomsCalculators.forces(sys, calc_ace)))
      F2 = ustrip.(u"eV/Å", reduce(hcat, AtomsCalculators.forces(sys, calc_et)))
      V1 = ustrip.(u"eV", AtomsCalculators.virial(sys, calc_ace))
      V2 = ustrip.(u"eV", AtomsCalculators.virial(sys, calc_et))
      errs = (dE = max(errs.dE, abs(E1 - E2) / nat),
              dF = max(errs.dF, maximum(abs.(F1 .- F2))),
              dV = max(errs.dV, maximum(abs.(V1 .- V2)) / nat))
      @test abs(E1) > 1e-3          # the pair term is live, so the gate is not vacuous
   end
   @info "ace1 pair conversion" errs
   return errs
end

@testset "ETPairModel from an ace1_model pair term (ACE1_PolyEnvelope1sR)" begin

   twin = ace1_exact_twin(; ACE1P_KW...)
   spl = ace1_model(; ACE1P_KW...).model

   @testset "the twin is ace1_model's basis" begin
      @test ace1_twin_matches(twin, spl)
      envs = twin.pairbasis.envelopes
      @test all(e isa M.ACE1_PolyEnvelope1sR for e in envs)
      # per-pair r0, so the per-pair lookup is observable
      @test length(unique(e.r0 for e in envs)) == 6
   end

   @testset "ACEModel pair energy / forces / virial, 1e-12" begin
      ps, st = ace1p_pair_only(twin)
      errs = ace1p_compare(twin, ps, st)
      @test errs.dE <= 1e-12
      @test errs.dF <= 1e-12
      @test errs.dV <= 1e-12
   end

   @testset "asymmetric (ordered-pair) envelope, 1e-12" begin
      # ace1_model's envelopes are symmetric in (iz, jz), which cannot tell an ordered
      # lookup from a transposed one.  Make them asymmetric: the ACEModel reads
      # envelopes[iz0, jz] (centre first), and so must the conversion.
      pb = twin.pairbasis
      NZ = length(pb._i2z)
      envs = [M.ACE1_PolyEnvelope1sR(pb.envelopes[i, j].rcut,
                                     pb.envelopes[i, j].r0 * (1 + 0.07 * i - 0.04 * j),
                                     pb.envelopes[i, j].p) for i in 1:NZ, j in 1:NZ]
      pb_asym = M.LearnableRnlrzzBasis(pb._i2z, pb.polys, pb.transforms,
                                       SMatrix{NZ, NZ}(envs), pb.rin0cuts, pb.spec; Winit = :onehot)
      @test any(pb_asym.envelopes[i, j].r0 != pb_asym.envelopes[j, i].r0
                for i in 1:NZ, j in 1:NZ)
      model_a = M.ace_model(; elements = ACE1P_ELEMENTS, order = 2, Ytype = :spherical,
                              ZBL = false, E0s = nothing, rbasis = twin.rbasis,
                              pair_basis = pb_asym, rin0cuts = twin.rbasis.rin0cuts,
                              level = ACEpotentials.ACE1compat._get_degrees(
                                         ACEpotentials.ACE1compat._clean_args(ACE1P_KW))[1],
                              max_level = ACEpotentials.ACE1compat._get_degrees(
                                         ACEpotentials.ACE1compat._clean_args(ACE1P_KW))[2],
                              init_WB = :zeros)
      ps, st = ace1p_pair_only(model_a)
      errs = ace1p_compare(model_a, ps, st)
      @test errs.dE <= 1e-12
      @test errs.dF <= 1e-12
      @test errs.dV <= 1e-12
   end

   @testset "an unsupported pair envelope is refused by name" begin
      # PolyEnvelope2sX is what ace1_model's pair basis uses with ZBL = true.
      pb = twin.pairbasis
      NZ = length(pb._i2z)
      envs = [M.PolyEnvelope2sX(-1.0, 1.0, 0, 2) for i in 1:NZ, j in 1:NZ]
      pb_x = M.LearnableRnlrzzBasis(pb._i2z, pb.polys, pb.transforms,
                                    SMatrix{NZ, NZ}(envs), pb.rin0cuts, pb.spec; Winit = :onehot)
      err = try
         ETM._convert_pair_envelope(pb_x.envelopes); nothing
      catch e
         e
      end
      @test err isa ErrorException
      @test err !== nothing && occursin("PolyEnvelope2sX", sprint(showerror, err))
   end
end
