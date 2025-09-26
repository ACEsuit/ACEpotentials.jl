# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Random, Zygote, Optimisers

# JuLIP (via ACEpotentials) also exports the same functions 
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
# we probably have to put significant work into initializing better.

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

@info("Subset of radial basis specification")
display(model.rbasis.spec[1:10:end])
@info("Subset of pair basis specification")
display(model.pairbasis.spec[1:2:end])

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

# From the model we generate a calculator. This step should probably be 
# integrated into `ace_model`, we can discuss it. 

calc = M.ACEPotential(model)

# We can now treat `calc` as a nonlinear parameterized site potential model. 
# - generate a random Al, Ti structure  
# - calculate the energy, forces, and virial
# An important point to note is that AtomsBase enforces the use of units. 

function rand_AlTi(nrep, rattle)
   # Al : 13; Ti : 22
   at = rattle!(bulk(:Al, cubic=true) * nrep, 0.1)
   # swap random atoms to Ti
   particles = map( enumerate(at) ) do (i, atom)
      (rand() < 0.5) ? AtomsBase.Atom(22, position(atom)) : atom
   end
   return FlexibleSystem(particles, cell_vectors(at), boundary_conditions(at))      
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
data = FlexibleSystem.(data)
train_data = data[1:5:end]
test_data = data[2:5:end]

# to set up training we specify data keys and training weights. 
# to get a unitless loss we need to specify the weights to have inverse 
# units to the data. The local loss function is the loss applied to 
# a single training structure. This follows the Lux training API 
#      loss(model, ps, st, data)

data_keys = (E_key = :energy, F_key = :force, V_key = :virial) 
weights = (wE = 1.0/u"eV", wF = 0.1 / u"eV/Å", wV = 0.1/u"eV")

loss = let data_keys = data_keys, weights = weights

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

at1 = train_data[1]
loss(calc, ps, st, at1)[1]


# Zygote should now be able to differentiate this loss with respect to parameters 
# the gradient is provided in the same format as the parameters, i.e. a NamedTuple. 
 
g = Zygote.gradient(ps -> loss(calc, ps, st, at1)[1], ps)[1] 

@show typeof(g)

# both parameters and gradients can be serialized into a vector and that 
# allows us use of arbitrary optimizers 

ps_vec, _restruct = destructure(ps)
g_vec, _ = destructure(g)   # the restructure is the same as for the params

# Let's now try to optimize the model. Here I'm a bit hazy how to do this 
# properly. 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the Lux approach, which I couldn't get to work. 
# I'm just modifying a Lux tutorial. There are probably better ways. 
# https://github.com/LuxDL/Lux.jl/blob/main/examples/PolynomialFitting/main.jl

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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# the alternative might be to optimize using Optim.jl
# this would allow us to use a wide range of different optimizers, 
# including BFGS and friends. 
# the total_loss and total_loss_grad! should be re-implemented properly 
# with options to assemble the loss multi-threaded or distributed 

using Optim

function total_loss(p_vec) 
   return sum( loss(calc, _restruct(p_vec), st, at)[1] 
                           for at in train_data )
end
                              
function total_loss_grad!(g, p_vec) 
   g[:] = Zygote.gradient(ps -> total_loss(ps), p_vec)[1]
   return g 
end 

@info("Timing for total loss and loss-grad")
@time total_loss(ps_vec)
@time total_loss(ps_vec)
@time total_loss_grad!(zeros(length(ps_vec)), ps_vec)
@time total_loss_grad!(zeros(length(ps_vec)), ps_vec)

@info("Start the optimization")
result = Optim.optimize(total_loss, total_loss_grad!, ps_vec;
                        method = Optim.Adam(),
                        show_trace = true, 
                        iterations = 30)   # obviously this needs more iterations


# We didn't use enough iterations to do anything useful here. But suppose we 
# had converged the nonlinear optimization to get a good radial basis. 
# Then, in a second step we can freeze the radial basis and 
# optimize the ACE basis coefficients via linear regression. 

# as a first step, we replace the learnable radials with 
# splined radials. This is not technically needed, but I want to make it 
# the default that once we have fixed the radials, we splinify them 
# so that we fit to exactly what we export.

ps1 = _restruct(result.minimizer)
lin_calc = M.splinify(calc, ps1)
lin_ps, lin_st = Lux.setup(rng, lin_calc)

# The next point is that I propose a change to the interface for evaluating 
# the basis (as opposed to the model), i.e. replacing 
#  energy(basis) with energy_basis(model)  and similar. 
# With this in mind we can now assemble the linear regression problem.

function local_lsqsys(calc, at, ps, st, weights, keys) 
   efv = M.energy_forces_virial_basis(at, calc, ps, st)

   # compute the E0s contribution. This needs to be done more 
   # elegantly and a stacked model would solve this problem. 
   E0 = sum( calc.model.E0s[M._z2i(calc.model, z)] 
             for z in AtomsBase.atomic_number(at) ) * u"eV"

   # energy 
   wE = weights[:wE]
   E_dft = at.data[data_keys.E_key] * u"eV"
   y_E = wE * (E_dft - E0) 
   A_E = wE * efv.energy' 

   # forces 
   wF = weights[:wF]
   F_dft = at.data[data_keys.F_key] * u"eV/Å"
   y_F = wF * reinterpret(eltype(F_dft[1]), F_dft)
   A_F = wF * reinterpret(eltype(efv.forces[1]), efv.forces)

   # virial 
   wV = weights[:wV]
   V_dft = at.data[data_keys.V_key] * u"eV"
   y_V = wV * V_dft[:]
   # display( reinterpret(eltype(efv.virial), efv.virial) )
   A_V = wV * reshape(reinterpret(eltype(efv.virial[1]), efv.virial), 9, :)

   return vcat(A_E, A_F, A_V), vcat(y_E, y_F, y_V)
end


# this line just checks that the local assembly makes sense 

A1, y1 = local_lsqsys(lin_calc, at1, lin_ps, lin_st, weights, data_keys)

@assert size(A1, 1) == length(y1) == 1 + 3 * length(at1) + 9
@assert size(A1, 2) == length(destructure(lin_ps)[1])

# we convert this to a global assembly routine. I thought this version would 
# be multi-threaded but something seems to be wrong with it. 
# this assembly is very slow because the current implementation of the 
# basis is very inefficient 

using Folds 

function assemble_lsq(calc, data, weights, data_keys; 
                      rng = Random.GLOBAL_RNG, 
                      executor = Folds.ThreadedEx())
   ps, st = Lux.setup(rng, calc)
   blocks = Folds.map(at -> local_lsqsys(lin_calc, at, ps, st, 
                                   weights, data_keys), 
                      train_data, executor)
   A = reduce(vcat, [b[1] for b in blocks])
   y = reduce(vcat, [b[2] for b in blocks])
   return A, y
end

A, y = assemble_lsq(lin_calc, train_data, weights, data_keys)

# estimate the parameters 

solver = ACEpotentials.ACEfit.BLR()
result = ACEpotentials.ACEfit.solve(solver, A, y)

# a little hack to turn it into NamedTuple parameters and a fully parameterized 
# model this needs another convenience function provided within ACEpotentials. 

ps, st = Lux.setup(rng, lin_calc)
_, _restruct = destructure(ps)
fit_ps = _restruct(result["C"])
fit_calc = M.ACEPotential(lin_calc.model, fit_ps, st)

# can this do anything useful? 
# first of all, because we have now specified the parameters, we no longer need 
# to drag them around and can use a higher-level interface to evaluate the 
# model. For example ... 
# (this should really go into unit tests I think)

using AtomsCalculators
using AtomsCalculators: energy_forces_virial, forces, potential_energy, virial 

at1 = rand(train_data)
efv1 = M.energy_forces_virial(at1, fit_calc, fit_calc.ps, fit_calc.st)
efv2 = energy_forces_virial(at1, fit_calc)
efv1.energy ≈ efv2.energy
all(efv1.forces .≈ efv2.forces)
efv1.virial ≈ efv2.virial
potential_energy(at1, fit_calc) ≈ efv1.energy
virial(at1, fit_calc) ≈ efv1.virial
ef = AtomsCalculators.energy_forces(at1, fit_calc)
ef.energy ≈ efv1.energy
all(ef.forces .≈ efv1.forces)
all(efv1.forces .≈ forces(at1, fit_calc))

# Checking accuracy (it is terrible, so lots to fix ...)

E_err(at) = abs(ustrip(potential_energy(at, fit_calc)) - at.data[data_keys.E_key]) / length(at)
mae_train = sum(E_err, train_data) / length(train_data)
mae_test = sum(E_err, test_data) / length(test_data)

@info("MAE(train) = $mae_train")
@info("MAE(test) = $mae_test")


# Trying some simple geometry optimization 

using GeomOpt

at = rand_AlTi(2, 0.001)
@show potential_energy(at, fit_calc)
_at_opt, info = GeomOpt.minimise(at, fit_calc; g_tol = 1e-4, g_calls_limit = 30 )
@show potential_energy(_at_opt, fit_calc)


# The last step is to run a simple MD simulation for just a 100 steps.
# Important: Tell Molly what units are used!!

import Molly 
at = rand_AlTi(3, 0.01)
sys_md = Molly.System(at; force_units=u"eV/Å", energy_units=u"eV")
temp = 298.0u"K"

sys_md = Molly.System(
   sys_md;
   general_inters = (fit_calc,),
   velocities = Molly.random_velocities(sys_md, temp),
   loggers=(temp=Molly.TemperatureLogger(100),)
)

simulator = Molly.VelocityVerlet(
   dt = 1.0u"fs",
   coupling = Molly.AndersenThermostat(temp, 1.0u"ps"),
)

Molly.simulate!(sys_md, simulator, 100)

@info("This simulation obviously crashed:")
@show sys_md.loggers.temp.history
