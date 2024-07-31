# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using Random
using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf 
rng = Random.GLOBAL_RNG
M = ACEpotentials.Models

include(@__DIR__() * "/LLSQ.jl")

##
# we will try this for a simple dataset, Zuo et al 
# replace element with any of those available in that dataset 

Z0 = :Si
train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")
train = train[1:3:end]
wE = 30.0; wF = 1.0; wV = 1.0 

##
# First we create an ACE1 style potential with some standard parameters 

elements = [Z0,]
order = 3 
totaldegree = 12

model2 = M.ace_model(; elements = elements, 
                       order = order,               # correlation order 
                       Ytype = :spherical,              # solid vs spherical harmonics
                       level = M.TotalDegree(),     # how to calculate the weights to give to a basis function
                       max_level = totaldegree+1,     # maximum level of the basis functions
                       pair_maxn = totaldegree,     # maximum number of basis functions for the pair potential 
                       init_WB = :zeros,            # how to initialize the ACE basis parmeters
                       init_Wpair = "linear",         # how to initialize the pair potential parameters
                       init_Wradial = :linear, 
                       pair_transform = (:agnesi, 1, 3), 
                       pair_learnable = false, 
                     )


# wrap the model into a calculator, which turns it into a potential...

ps, st = Lux.setup(rng, model2)
calc_model2 = M.ACEPotential(model2, ps, st)


##
# Fit the ACE2 model - this still needs a bit of hacking to convert everything 
# to the new framework. 
# - convert the data to AtomsBase 
# - use a different interface to specify data weights and keys 
#   (this needs to be brough in line with the ACEpotentials framework)
# - rewrite the assembly for the LSQ system from scratch (but this is easy)

train2 = FlexibleSystem.(train)
test2 = FlexibleSystem.(test)
data_keys = (E_key = :energy, F_key = :force, ) 
weights = (wE = wE/u"eV", wF = wF / u"eV/Å", )

A, y = LLSQ.assemble_lsq(calc_model2, train2, weights, data_keys)
@show size(A) 

θ = ACEfit.trunc_svd(svd(A), y, 1e-8)
calc_model2_fit = LLSQ.set_linear_params(calc_model2, θ)

##
# Look at errors

E_rmse_2, F_rmse_2 = LLSQ.rmse(test2, calc_model2_fit)

@printf("Model  |      E    |    F  \n")
@printf(" ACE2  | %.2e  |  %.2e  \n", E_rmse_2, F_rmse_2)

