
module ACE1_TestUtils 

using Random, Test, ACEbase, LinearAlgebra, Lux
using ACEbase.Testing: print_tf, println_slim
import ACE1, ACE1x, JuLIP

using ACEpotentials 
M = ACEpotentials.Models 
ACE1compat = ACEpotentials.ACE1compat
rng = Random.MersenneTwister(1234)

function JuLIP.cutoff(model::ACE1x.ACE1Model)
   return maximum(JuLIP.cutoff.(model.basis.BB)) 
end

_evaluate(basis::JuLIP.MLIPs.IPSuperBasis, Rs, Zs, z0) = 
      reduce(vcat, [ACE1.evaluate(B, Rs, Zs, z0) for B in basis.BB])


JuLIP.evaluate(V::JuLIP.OneBody, Rs, Zs, z0) = 
      V.E0[JuLIP.Chemistry.chemical_symbol(z0)]

_evaluate(pot::JuLIP.MLIPs.SumIP, Rs, Zs, z0) = 
      sum(JuLIP.evaluate(V, Rs, Zs, z0) for V in pot.components)

function get_rbasis(model::ACE1x.ACE1Model)
   return model.basis.BB[2].pibasis.basis1p.J
end

function get_rbasis(model::M.ACEModel)
   return model.rbasis
end

function check_rbasis_transforms(model1, model2)
   @info("check the radial basis transforms and envelopes")

   NZ = M._get_nz(model2); @assert NZ == 1

   rbasis1 = get_rbasis(model1)
   rbasis2 = get_rbasis(model2)
   
   trans1 = rbasis1.trans.transforms[1]
   trans2 = rbasis2.transforms[1]
   
   rcut = JuLIP.cutoff(model1)
   rr = rcut * rand(200)
   t1 = ACE1.Transforms.transform.(Ref(trans1), rr)
   t2 = trans2.(rr)
   err_t1_t2 = maximum(abs.(t1 .- t2))
   println_slim(@test err_t1_t2 < 1e-12)
   
   # the envelope - check that the "choices" are the same 
   println_slim(@test rbasis1.envelope isa ACE1.OrthPolys.OneEnvelope)
   println_slim(@test rbasis1.J.pl == rbasis1.J.pr == 2 )
   println_slim(@test rbasis2.envelopes[1].p1 == rbasis2.envelopes[1].p2 == 2)
   nothing 
end 


function check_rbasis(model1, model2) 
   @info("check radial basis")
   @info("    error can be a bit larger since the jacobi basis used in ACE1 is constructed from a discrete measure")
   @info("The first test checks Rn vs Rn0")
   z1 = AtomicNumber(:Si)
   z2 = Int(z1)
   rbasis1 = get_rbasis(model1)
   rbasis2 = get_rbasis(model2)
   rcut = JuLIP.cutoff(model1) 
   rp = range(0.0, rcut, length=200)
   R1 = reduce(hcat, [ ACE1.evaluate(rbasis1, r, z1, z1) for r in rp ])
   R2 = reduce(hcat, [ rbasis2(r, z2, z2, NamedTuple(), NamedTuple()) for r in rp])
   maxn = size(R1, 1)
   scal = [ maximum(R1[n,:]) / maximum(R2[n,:]) for n = 1:maxn ] 
   errs = maximum(abs, R1 - Diagonal(scal) * R2[1:maxn, :]; dims=2)
   normalizederr = norm(errs ./ (1:length(errs)).^3, Inf)
   @show normalizederr 
   println_slim(@test normalizederr < 1e-4)
   
   @info("The remaining checks are for Rn0 = Rnl")
   for i_nl = 1:size(R2, 1)
      n = rbasis2.spec[i_nl].n 
      print_tf(@test R2[i_nl, :] ≈ R2[n, :])
   end
   println()
end


function check_pairbasis(model1, model2) 
   @info("Check the pair basis")
   pairbasis1 = model1.basis.BB[1]
   pairbasis2 = model2.pairbasis
   rcut = JuLIP.cutoff(model1)
   z1 = AtomicNumber(:Si)
   z2 = Int(z1)
   rr = range(0.001, rcut, length=200)
   P1 = reduce(hcat, [ ACE1.evaluate(pairbasis1, r, z1, z1) for r in rr ])
   P2 = reduce(hcat, [ pairbasis2(r, z2, z2, NamedTuple(), NamedTuple()) for r in rr])
   println_slim(@test size(P1) == size(P2))
   
   nmax = size(P1, 1)
   scal_pair = [ sum(P1[n, 70:end]) / sum(P2[n, 70:end]) for n = 1:nmax ]
   P2 = Diagonal(scal_pair) * P2
   scal_err = abs( -(extrema(abs.(scal_pair))...) )
   @show scal_err
   println_slim(@test scal_err < 0.01)
   
   err = maximum(abs, (P1 - P2) ./ (abs.(P1) .+ abs.(P2) .+ 1); dims=2)
   normalizederr = norm(err ./ (1:length(err)).^3, Inf)
   @show normalizederr
   println_slim(@test normalizederr < 1e-3)
end 


function check_basis(model1, model2; Nenv = :auto) 
   ps, st = Lux.setup(rng, model2)

   @info("Check the bases span the same space")
   NZ = M._get_nz(model2); @assert NZ == 1
   if NZ == 1
      @info("   NZ == 1  >>>  check spec matches")
      _spec1 = ACE1.get_nl(model1.basis.BB[2])
      spec1 = [ [ (n = b.n, l = b.l) for b in bb ] for bb in _spec1 ]
      spec2 = M.get_nnll_spec(model2.tensor)
      println_slim(@test issubset(sort.(spec1), sort.(spec2))) 
   end 
   
   lenB1 = length(model1.basis)
   if Nenv == :auto 
      Nenv = lenB1 * 10 
      @show Nenv 
   end 

   Random.seed!(12345)
   XX2 = [ M.rand_atenv(model2, rand(6:10)) for _=1:Nenv ]
   XX1 = [ (x[1], AtomicNumber.(x[2]), AtomicNumber(x[3])) for x in XX2 ]
   
   B1 = reduce(hcat, [ _evaluate(model1.basis, x...) for x in XX1])
   B2 = reduce(hcat, [ M.evaluate_basis(model2, x..., ps, st) for x in XX2])
   @info("Compute linear transform between bases to show match") 
   # We want full-rank C such that C * B2 = B1 
   # (note this allows B2 > B1)
   C = B2' \ B1'
   basiserr = norm(B1 - C' * B2, Inf)
   @show basiserr
   println_slim(@test basiserr < .3e-2)
   
   # # some more fine-grained checks for debugging 
   # Nmb = length(spec1)
   # B1_mb = B1[end-Nmb+1:end, :]
   # B2_mb = B2[1:Nmb, :]
   # 
   # C_mb = B1_mb' \ B2_mb'
   # @show norm(B2_mb - C_mb' * B1_mb, Inf)
   
   # B1_pair = B1[1:end-Nmb, :]
   # B2_pair = B2[Nmb+1:end, :]
   # C_pair = B1_pair' \ B2_pair'
   # @show norm(B2_pair - C_pair' * B1_pair, Inf)

   @info("Set some random parameters and check site energies")
   θ1 = randn(lenB1) ./ (1:lenB1).^2
   θ2 = C * θ1
   
   ACE1x._set_params!(model1, θ1)
   
   calc2 = M.ACEPotential(model2, ps, st)
   M.set_parameters!(calc2, θ2)
   
   V1 = [ _evaluate(model1.potential, x...) for x in XX1 ]
   V2 = [ M.evaluate(calc2.model, x..., calc2.ps, calc2.st) for x in XX2 ]
   
   err = norm(V1 - V2, Inf)
   @show err
   println_slim(@test err < 1e-4)
   nothing 
end 


function check_compat(params; deginc = 0.1) 
   model1 = acemodel(; params...)
   params2 = (; params..., totaldegree = params.totaldegree .+ deginc)
   model2 = ACE1compat.ace1_model(; params...)

   NZ = length(params.elements)
   if NZ == 1
      @info("NZ == 1  >>>  can do some extra checks")
      check_rbasis_transforms(model1, model2)
      check_rbasis(model1, model2)
      check_pairbasis(model1, model2)
   end 
   
   check_basis(model1, model2)
   nothing 
end 

end