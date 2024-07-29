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
                       init_Wpair = "linear",         # how to initialize the pair potential parameters
                       init_Wradial = :linear, 
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
k = length(rbasis1.J.A)

# transform old coefficients to new coefficients to make them match 

rbasis1.J.A[:] .= rbasis2.polys.A[1:k]
rbasis1.J.B[:] .= rbasis2.polys.B[1:k]
rbasis1.J.C[:] .= rbasis2.polys.C[1:k]
rbasis1.J.A[2] /= rbasis1.J.A[1] 
rbasis1.J.B[2] /= rbasis1.J.A[1]

# wrap the model into a calculator, which turns it into a potential...

calc_model2 = M.ACEPotential(model2)


##

ybasis1 = model1.basis.BB[2].pibasis.basis1p.SH
ybasis2 = model2.ybasis
maxk = length(ybasis2)

X = [ (u = @SVector rand(3); u/norm(u)) for _ = 1:100 ]
Y1 = reduce(hcat, [ ACE1.evaluate(ybasis1, u)[1:maxk] for u in X ])
Y1r = real.(Y1) 
Y1i = imag.(Y1)
Y2 = reduce(hcat, [ ybasis2(u)[1:maxk] for u in X ])

@info("check span real/imag(Y1) = span Y2")
Cr = Y2' \ Y1r' 
@show norm(Y1r' - Y2' * Cr)

Ci = Y2' \ Y1i' 
@show norm(Y1i' - Y2' * Ci)


## 

cyp4ml = complex_sphericalharmonics(2)
Yp4 = reduce(hcat, [cyp4ml(u) for u in X] )
Yp4 ≈ Y1
