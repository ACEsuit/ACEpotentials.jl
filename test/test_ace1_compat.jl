
using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
# using TestEnv; TestEnv.activate();

##

using Plots
using Random, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models
ACE1compat = ACEpotentials.ACE1compat

##



params = ( elements = [:Si,], 
           order = 3, 
           transform = (:agnesi, 2, 2),
           totaldegree = 8, 
           pure = false, 
           pure2b = false,
           pair_envelope = (:r, 1),
           rcut = 5.5,
         )


model1 = acemodel(; params...)


## 

@info("check the transform construction")

params_clean = ACE1compat._clean_args(params)
rbasis1 = model1.basis.BB[2].pibasis.basis1p.J
rbasis2 = ACE1compat._radial_basis(params_clean)

trans1 = rbasis1.trans.transforms[1]
trans2 = rbasis2.transforms[1]

rr = params.rcut * rand(200)
t1 = ACE1.Transforms.transform.(Ref(trans1), rr)
t2 = trans2.(rr)
err_t1_t2 = maximum(abs.(t1 .- t2))
println_slim(@test err_t1_t2 < 1e-12)

# the envelope - check that the "choices" are the same 

println_slim(@test rbasis1.envelope isa ACE1.OrthPolys.OneEnvelope)
println_slim(@test rbasis1.J.pl == rbasis1.J.pr == 2 )
println_slim(@test rbasis2.envelopes[1].p1 == rbasis2.envelopes[1].p2 == 2)

##

@info("check full radial basis construction")
@info("    This error can be a bit larger since the jacobi basis used in ACE1 is constructed from a discrete measure")
@info("The first test checks Rn vs Rn0")
z1 = AtomicNumber(:Si)
z2 = Int(z1)
rp = range(0.0, params.rcut, length=200)
R1 = reduce(hcat, [ ACE1.evaluate(rbasis1, r, z1, z1) for r in rp ])
R2 = reduce(hcat, [ rbasis2(r, z2, z2, NamedTuple(), NamedTuple()) for r in rp])
maxn = size(R1, 1)
scal = [ maximum(R1[n,:]) / maximum(R2[n,:]) for n = 1:maxn ] 
err = norm(R1 - Diagonal(scal) * R2[1:maxn, :], Inf)
@show err 
println_slim(@test err < 0.001)

@info("The remaining checks are for Rn0 = Rnl")
for i_nl = 1:size(R2, 1)
   n = rbasis2.spec[i_nl].n 
   print_tf(@test R2[i_nl, :] ≈ R2[n, :])
end
println()

## 

@info("Check the pair basis construction")
pairbasis1 = model1.basis.BB[1]
pairbasis2 = ACE1compat._pair_basis(params_clean)

rr = range(0.001, params.rcut, length=200)
P1 = reduce(hcat, [ ACE1.evaluate(pairbasis1, r, z1, z1) for r in rr ])
P2 = reduce(hcat, [ pairbasis2(r, z2, z2, NamedTuple(), NamedTuple()) for r in rr])
println_slim(@test size(P1) == size(P2))

nmax = size(P1, 1)
scal = [ sum(P1[n, 70:end]) / sum(P2[n, 70:end]) for n = 1:nmax ]
P2 = Diagonal(scal) * P2

err = norm( (P1 - P2) ./ (abs.(P1) .+ abs.(P2) .+ 1), Inf)
@show err 
println_slim(@test err < 0.01)


