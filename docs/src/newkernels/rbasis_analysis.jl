# This script is to explore the differences between the ACE1 models and the new 
# models. This is to help bring the two to feature parity so that ACE1 
# can be retired. 

using Random
using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Plots
rng = Random.GLOBAL_RNG

# we will try this for a simple dataset, Zuo et al 
# replace element with any of those available in that dataset 

Z0 = :Si 
z1 = AtomicNumber(Z0)
z2 = Int(z1)

train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")
train = train[1:3:end]

# because the new implementation is experimental, it is not exported, 
# so I create a little shortcut to have easy access. 

M = ACEpotentials.Models

# First we create an ACE1 style potential with some standard parameters 

elements = [Z0,]
order = 3 
totaldegree = 10
rcut = 5.5 

model1 = acemodel(elements = elements, 
                  order = order, 
                  transform = (:agnesi, 2, 2),
                  totaldegree = totaldegree, 
                  pure = false, 
                  pure2b = false, 
                  pair_envelope = (:r, 1), 
                  rcut = rcut,  )

# now we create an ACE2 style model that should behave similarly                   

# this essentially reproduces the rcut = 5.5, we may want a nicer way to 
# achieve this. 

rin0cuts = M._default_rin0cuts(elements) #; rcutfactor = 2.29167)
rin0cuts = SMatrix{1,1}((;rin0cuts[1]..., :rcut => 5.5))

model2 = M.ace_model(; elements = elements, 
                       order = order,               # correlation order 
                       Ytype = :solid,              # solid vs spherical harmonics
                       level = M.TotalDegree(),     # how to calculate the weights to give to a basis function
                       max_level = totaldegree,     # maximum level of the basis functions
                       pair_maxn = totaldegree,     # maximum number of basis functions for the pair potential 
                       init_WB = :zeros,            # how to initialize the ACE basis parmeters
                       init_Wpair = :onehot,         # how to initialize the pair potential parameters
                       init_Wradial = :onehot, 
                       pair_transform = (:agnesi, 1, 3), 
                       pair_learnable = true, 
                       rin0cuts = rin0cuts, 
                     )

ps, st = Lux.setup(rng, model2)                     
ps_r = ps.rbasis
st_r = st.rbasis

# extract the radial basis 
rbasis1 = model1.basis.BB[2].pibasis.basis1p.J
rbasis2 = model2.rbasis
k = length(rbasis1_.J.A)

# transform old coefficients to new coefficients to make them match 

rbasis1.J.A[:] .= rbasis2.polys.A[1:k]
rbasis1.J.B[:] .= rbasis2.polys.B[1:k]
rbasis1.J.C[:] .= rbasis2.polys.C[1:k]
rbasis1.J.A[2] /= rbasis1.J.A[1] 
rbasis1.J.B[2] /= rbasis1.J.A[1]

# wrap the model into a calculator, which turns it into a potential...

calc_model2 = M.ACEPotential(model2)


##

# sample points
rr = range(0.001, rcut, length=200)

@info("check the transforms are identical") 
t1 = ACE1.Transforms.transform.(Ref(rbasis1.trans), rr, z1, z1)
t2 = rbasis2.transforms[1].(rr)
@show t1 ≈ t2

##

@info("Check the raw polynomials")
xx = range(-1, 1, length=200)
J1 = reduce(hcat, ACE1.evaluate.(Ref(_J), xx))' 
J2 = rbasis2.polys(xx)
J1 ≈ J2[:, 1:size(J1, 2)]

plt = plot() 
for n = 1:5 
   plot!(J1[:, n], c = n, label = "J1,$n")
   plot!(J2[:, n], c = n, ls = :dash, label = "J2,$n")
end
plt

##

R1 = reduce(hcat, [ JuLIP.evaluate(rbasis1, r, z1, z1) for r in rr ])
R2 = reduce(hcat, [ rbasis2(r, z2, z2, ps_r, st_r)[1:10] for r in rr])

# R1 = R1_ * Diagonal(R2[1,:])

# normalize 
for n = 1:10 
   R1[n, :] = R1[n, :] / maximum(abs, R1[n, :])
   R2[n, :] = R2[n, :] / maximum(abs, R2[n, :])
end

plt = plot() 
for n = 1:4
   plot!(plt, rr, R1[n, :], c = n, label="R1_$n")
   plot!(plt, rr, R2[n, :], c = n, ls = :dash, label="R2_$n")
end
plt


##

pairb1 = model1.basis.BB[1].J[1] 
pairb2 = model2.pairbasis 
ps_pair = ps.pairbasis
st_pair = st.pairbasis

P1 = reduce(hcat, [ JuLIP.evaluate(pairb1, r, z1, z1)[1:nmax] for r in rr ])
P2 = reduce(hcat, [ pairb2(r, z2, z2, ps_pair, st_pair)[1:nmax] for r in rr ])

# truncate 
P1 = min.(max.(P1, -100), 100)
P2 = min.(max.(P2, -100), 100)

plt = plot(; ylims = (-1.0, 3.0)) 
for n = 1:4
   plot!(plt, rr, P1[n, :], c = n, label="P1_$n")
   plot!(plt, rr, P2[n, :], c = n, ls = :dash, label="P2_$n")
end
plt


##
@info(" Confirm that the pair bases span the same space ")
# they do to within almost machine precision
# ( in fact the transormation matrix C says they differ only 
#   up to a sign and scalar factor. )

rrr = range(1.0, 5.0, length=100)
P1 = reduce(hcat, [ JuLIP.evaluate(pairb1, r, z1, z1)[1:nmax] for r in rrr ])
P2 = reduce(hcat, [ pairb2(r, z2, z2, ps_pair, st_pair)[1:nmax] for r in rrr ])
C = P2' \ P1'
@show norm(P1' - P2' * C)

@info("Confirm the radial bases span the same space")

R1 = reduce(hcat, [ JuLIP.evaluate(rbasis1, r, z1, z1)[1:nmax] for r in rrr ])
R2 = reduce(hcat, [ rbasis2(r, z2, z2, ps_r, st_r)[1:nmax] for r in rrr])
C = R2' \ R1'
@show norm(R1' - R2' * C)

