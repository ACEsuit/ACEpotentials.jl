# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using Random
using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf 
# rng = Random.GLOBAL_RNG
# M = ACEpotentials.Models

include("match_bases.jl")
include("llsq.jl")

# we will try this for a simple dataset, Zuo et al 
# replace element with any of those available in that dataset 

Z0 = :Si
train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")
train = train[1:3:end]
weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),);

# Creating matching ACE1 and ACE2 models 

totaldegree = 10
order = 3 
rcut = 5.5 

model1, model2, calc_model2 = matching_bases(; 
      Z = Z0, rcut = rcut, order = order, totaldegree=totaldegree)


## 
#Fit the ACE1 model 

solver=ACEfit.TruncatedSVD(; rtol = 1e-8)
acefit!(model1, train;  solver=solver, weights=weights)


##
# Fit the ACE2 model - this still needs a bit of hacking to convert everything 
# to the new framework. To be moved into ACEpotentials and hide from 
# the user ...  
# - convert the data to AtomsBase 
# - use a different interface to specify data weights and keys 
#   (this needs to be brough in line with the ACEpotentials framework)
# - rewrite the assembly for the LSQ system from scratch (but this is easy)

train2 = FlexibleSystem.(train)
test2 = FlexibleSystem.(test)
data_keys = (E_key = :energy, F_key = :force, ) 
weights = (wE = 30.0/u"eV", wF = 1.0 / u"eV/Å", )

A, y = LLSQ.assemble_lsq(calc_model2, train2, weights, data_keys)
@show size(A) 

θ = ACEfit.trunc_svd(svd(A), y, 1e-8)
calc_model2_fit = LLSQ.set_linear_params(calc_model2, θ)

##
# compute the errors 

E_rmse_1, F_rmse_1 = LLSQ.rmse(test, model1.potential)
E_rmse_2, F_rmse_2 = LLSQ.rmse(test2, calc_model2_fit)


@printf("Model  |     E    |    F  \n")
@printf(" ACE1  | %.2e  |  %.2e  \n", E_rmse_1, F_rmse_1)
@printf(" ACE2  | %.2e  |  %.2e  \n", E_rmse_2, F_rmse_2)

