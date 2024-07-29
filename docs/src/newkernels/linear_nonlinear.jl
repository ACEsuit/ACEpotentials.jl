# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using Random
using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf, Optim, LineSearches
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

E_train, F_train = LLSQ.rmse(train2, calc_model2_fit)
E_test, F_test = LLSQ.rmse(test2, calc_model2_fit)

@printf("       |      E    |    F  \n")
@printf(" train | %.2e  |  %.2e  \n", E_train, F_train)
@printf("  test | %.2e  |  %.2e  \n", E_test, F_test)


## 
# Now we can do some nonlinear iterations on the model 

# First we need to define a loss (dropping virials here...)

loss = let data_keys = data_keys, weights = weights

   function(calc, ps, st, at)
      efv = M.energy_forces_virial(at, calc, ps, st)
      _norm_sq(f) = sum(abs2, f)
      E_dft, F_dft = Zygote.ignore() do 
            (  at.data[data_keys.E_key] * u"eV", 
               at.data[data_keys.F_key] * u"eV/Å" )
      end
      return ( weights[:wE]^2 * (efv.energy - E_dft)^2 / length(at)  
               + weights[:wF]^2 * sum(_norm_sq, efv.forces - F_dft) 
            ), st, ()
   end
end


at1 = train2[1]
calc = deepcopy(calc_model2_fit)
loss(calc, calc.ps, calc.st, at1)

g = Zygote.gradient(ps -> loss(calc, ps, st, at1)[1], ps)[1] 


ps_vec, _restruct = destructure(calc.ps)

function total_loss(p_vec) 
   return sum( loss(calc, _restruct(p_vec), st, at)[1] 
                           for at in train2 )
end
                              
function total_loss_grad!(g, p_vec) 
   g[:] = Zygote.gradient(ps -> total_loss(ps), p_vec)[1]
   return g 
end 

total_loss_grad(p_vec) = total_loss_grad!(zeros(length(ps_vec)), ps_vec)

# these are reasonably fast - check with these: 
# @time total_loss(ps_vec)
# @time total_loss(ps_vec)
# @time total_loss_grad!(zeros(length(ps_vec)), ps_vec)
# @time total_loss_grad!(zeros(length(ps_vec)), ps_vec)

g = total_loss_grad!(zeros(length(ps_vec)), ps_vec)

@info("Start the optimization")
method = GradientDescent(; alphaguess = InitialHagerZhang(α0=1.0), 
                        linesearch = LineSearches.BackTracking(; order=2),)

result = Optim.optimize(total_loss, total_loss_grad!, ps_vec;
                        method = method,
                        show_trace = true, 
                        iterations = 30)   # obviously this needs more iterations

# this actually terminates unsuccessfull, without 
# progress. not sure why it says it was successful ... 

##
# we can try something purely manual ... 
ps_vec1 = deepcopy(ps_vec)

##

# the following look shows that it seems very very hard to 
# reduce the loss further. Unclear what is going on here... 

α = 1e-8 
for n = 1:10 
   l = total_loss(ps_vec1)
   g = total_loss_grad(ps_vec1)
   @printf(" %.2e | %.2e \n", l, norm(g, Inf))
   ps_vec1 -= α * g
end

## 

# trying Adam now? This is a bit randomized and it shows 
# that as soon as we perturb a bit, we get a much higher 
# loss and gradient. It suggests that the LLSQ picks out 
# an extremely unstable minimum.

method = Optim.Adam()
result = Optim.optimize(total_loss, total_loss_grad!, ps_vec;
                        method = method,
                        show_trace = true, 
                        iterations = 30)   # obviously this needs more iterations
