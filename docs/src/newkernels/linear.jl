# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers      
import Random: seed!, MersenneTwister      

# we will try this for a simple dataset, Zuo et al 
# replace element with any of those available in that dataset 

Z0 = :Si 
train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")

# because the new implementation is experimental, it is not exported, 
# so I create a little shortcut to have easy access. 

M = ACEpotentials.Models

# The new implementation tries to follow Lux rules, which likes to be 
# disciplined and explicit about random numbers 

rng = MersenneTwister(1234)

# First we create an ACE1 style potential with some standard parameters 

elements = (Z0,)
order = 3 
totaldegree = 6
rcut = 5.5 

@profview begin 
   model1 = acemodel(elements = elements, 
                  order = order, 
                  totaldegree = totaldegree, 
                  rcut = rcut,  )
end 

# now we create an ACE2 style model that should behave similarly                   

model2 = M.ace_model(; elements = elements, 
                      order = order,          # correlation order 
                      Ytype = :solid,     # solid vs spherical harmonics
                      level = M.TotalDegree(),   # how to calculate the weights to give to a basis function
                      max_level = totaldegree,     # maximum level of the basis functions
                      pair_maxn = totaldegree,     # maximum number of basis functions for the pair potential 
                      init_WB = :zeros,     # how to initialize the ACE basis parmeters
                      init_Wpair = :zeros,   # how to initialize the pair potential parameters
                      init_Wradial = :linear 
                      )

# Example dataset 

data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
data = FlexibleSystem.(data)
train_data = data[1:5:end]
test_data = data[2:5:end]
data_keys = (E_key = :energy, F_key = :force, V_key = :virial) 
weights = (wE = 1.0/u"eV", wF = 0.1 / u"eV/Å", wV = 0.1/u"eV")
