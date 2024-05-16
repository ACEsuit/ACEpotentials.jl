# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using ACEpotentials, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Random, Zygote, Optimisers

bulk = AtomsBuilder.bulk 
rattle! = AtomsBuilder.rattle!      

# because the new implementation is experimental, it is not exported, 
# so I create a little shortcut to have easy access. 

M = ACEpotentials.Models      

# The new implementation tries to follow Lux rules, which likes to be 
# disciplined and explicit about random numbers 

rng = Random.MersenneTwister(1234)


# I'll create a new model for a simple alloy and then generate a model. 
# this generates a trace-like ACE model with a random radial basis. 

elements = (:Al, :Ti)

model = M.ace_model(; elements = elements, 
                      order = 3,          # correlation order 
                      Ytype = :solid,     # solid vs spherical harmonics
                      level = M.TotalDegree(),   # how to calculate the weights to give to a basis function
                      max_level = 15,     # maximum level of the basis functions
                      pair_maxn = 15,     # maximum number of basis functions for the pair potential 
                      init_WB = :glorot_normal,     # how to initialize the ACE basis parmeters
                      init_Wpair = :glorot_normal   # how to initialize the pair potential parameters
                      )

# the radial basis specification can be looked at explicitly via 

display(model.rbasis.spec)

# we can see that it is defined as (n, l) pairs. Each `n` specifies an invariant 
# channel coupled to an `l` channel. Each `Rnl` radial basis function is defined 
# by `Rnl(r, Zi, Zj) = ∑_q W_nlq(Zi, Zj) * P_q(r)`. 

# some things that are missing: 
# - reweighting the basis via a smoothness prior. 
# - allow initialization of pair potential basis with one-hot embedding params 
#   right now the pair potential basis uses trace-like radials 
# - convenient ways to inspect the many-body basis specification. 

# Lux wants us to call a setup function to generate the parameters and state
# for the model.

ps, st = Lux.setup(rng, model)

# From the model we generate a calculator. This step should probably be integrated. 
# into `ace_model`, we can discuss it. 

calc = M.ACEPotential(model, ps, st)

# We can now treat `calc` as a nonlinear parameterized site potential model. 
# - generate a random Al, Ti structure  
# - calculate the energy, forces, and virial
# An important point to note is that AtomsBase enforces the use of units. 

function rand_AlTi(nrep, rattle)
   # Al : 13; Ti : 22
   at = rattle!(bulk(:Al, cubic=true) * 2, 0.1)
   Z = AtomsBuilder._get_atomic_numbers(at)
   Z[rand(1:length(at), length(at) ÷ 2)] .= 22
   return AtomsBuilder._set_atomic_numbers(at, Z)      
end


at = rand_AlTi(2, 0.1)

efv = M.energy_forces_virial(at, calc, ps, st)

@info("Energy")
display(efv.energy)

@info("Virial")
display(efv.virial)

@info("Forces (on atoms 1..5)")
display(efv.forces[1:5])

# we can incorporate the parameters and the state into the model struct 
# but for now we ignore this possibility and focus on how to train a model. 

# we load our example dataset and convert it to AtomsBase 

data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = FlexibleSystem.(data[1:5:end])

# to set up training we specify data keys and training weights. 
# to get a unitless loss we need to specify the weights to have inverse 
# units to the data. The local loss function is the loss applied to 
# a single training structure. This follows the Lux training API 
#      loss(model, ps, st, data)

loss = let data_keys = (E_key = :energy, F_key = :force, V_key = :virial), 
                   weights = (wE = 1.0/u"eV", wF = 0.1 / u"eV/Å", wV = 0.1/u"eV")

   function(calc, ps, st, at)
      efv = M.energy_forces_virial(at, calc, ps, st)
      _norm_sq(f) = sum(abs2, f)
      E_dft, F_dft, V_dft = Zygote.ignore() do   # Zygote doesn't have an adjoint for creating units :(
            (  at.data[data_keys.E_key] * u"eV", 
               at.data[data_keys.F_key] * u"eV/Å", 
               at.data[data_keys.V_key] * u"eV" )
      end
      return ( weights[:wE]^2 * (efv.energy - E_dft)^2 / length(at)  
               + weights[:wV]^2 * sum(abs2, efv.virial - V_dft) / length(at) 
               + weights[:wF]^2 * sum(_norm_sq, efv.forces - F_dft) 
            ), st, ()
   end
end    

loss(calc, ps, st, at)[1]


# Zygote should now be able to differentiate this loss with respect to parameters 
# the gradient is provided in the same format as the parameters, i.e. a NamedTuple. 
 
at1 = train_data[1] 
g = Zygote.gradient(ps -> loss(calc, ps, st, at1)[1], ps)[1] 

@show typeof(g)

# both parameters and gradients can be serialized into a vector and that 
# allows us use of arbitrary optimizers 

ps_vec, _restruct = destructure(ps)
g_vec = destructure(g)[1]

# Let's now try to optimize the model. Here I'm a bit hazy how to do this 
# properly. I'm just modifying a Lux tutorial. There are probably better ways. 
# https://github.com/LuxDL/Lux.jl/blob/main/examples/PolynomialFitting/main.jl

# This is the Lux approach, which I couldn't get to work. 

# using ADTypes, Printf 
# vjp_rule = AutoZygote() 
# opt = Optimisers.Adam()
# opt_state = Optimisers.setup(opt, ps)
# tstate = Lux.Experimental.TrainState(rng, calc.model, opt)

# function main(tstate, vjp, data, epochs)
#    for epoch in 1:epochs
#        grads, loss_val, stats, tstate = Lux.Experimental.compute_gradients(
#            vjp, loss, data, tstate)
#        if epoch % 10 == 1 || epoch == epochs
#            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss_val
#        end
#        tstate = Lux.Experimental.apply_gradients!(tstate, grads)
#    end
#    return tstate
# end

# main(tstate, vjp_rule, train_data, 100)


# the alternative might be to optimize using Optim.jl

using Optim

function total_loss(p_vec)
   return sum( loss(calc, _restruct(p_vec), st, at)[1] 
               for at in train_data )
end

function total_loss_grad!(g, p_vec) 
   g[:] = Zygote.gradient(ps -> total_loss(ps), p_vec)[1]
   return g 
end 

@time total_loss(ps_vec)
@time total_loss(ps_vec)
@time total_loss_grad!(zeros(length(ps_vec)), ps_vec)
@time total_loss_grad!(zeros(length(ps_vec)), ps_vec)

result = Optim.optimize(total_loss, total_loss_grad!, ps_vec;
                        method = Optim.Adam(),
                        show_trace = true, 
                        iterations = 100)


# Now that we've optimized the entire model a little bit 
# we can think that the radial basis functions are sufficiently 
# optimized. This is of course not true in this case since we didn't 
# use enough iterations. But suppose we had converged the nonlinear 
# optimization to get a really good radial basis. 
# Then, in a second step we can freeze the radial basis and 
# optimize the ACE basis coefficients via linear regression. 

# as a first step, we replace the learnable radials with 
# splined radials 

ps1_vec = result.minimizer
ps1 = _restruct(ps1_vec)

rbasis_p = M.set_params(calc.model.rbasis, ps1.rbasis)
rbasis_spl = M.splinify(rbasis_p)

# next we create a new ACE model with the splined radial basis
# this step should be moved into ACEpotentials.Models and 
# automated. 

linmodel = M.ACEModel(calc.model._i2z, 
               rbasis_spl, 
               calc.model.ybasis, 
               calc.model.abasis, 
               calc.model.aabasis, 
               calc.model.A2Bmap, 
               calc.model.bparams, 
               calc.model.pairbasis, 
               calc.model.pairparams, 
               calc.model.meta)
lincalc = M.ACEPotential(linmodel)

