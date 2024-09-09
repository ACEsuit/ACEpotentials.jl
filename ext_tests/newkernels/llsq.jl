
module LLSQ 

using Random, LinearAlgebra, Folds, Lux, Optimisers
using ACEpotentials
M = ACEpotentials.Models 
using Random: MersenneTwister


function local_lsqsys(calc, at, ps, st, weights, keys)
   efv = M.energy_forces_virial_basis(at, calc, ps, st)

   # There are no E0s in this dataset! 
   # # compute the E0s contribution. This needs to be done more 
   # # elegantly and a stacked model would solve this problem. 
   # E0 = sum( calc.model.E0s[M._z2i(calc.model, z)] 
   #           for z in AtomsBase.atomic_number(at) ) * u"eV"

   # energy 
   wE = weights[:wE]
   E_dft = at.data[keys.E_key] * u"eV"
   y_E = wE * E_dft / sqrt(length(at))   # (E_dft - E0)
   A_E = wE * efv.energy' / sqrt(length(at))

   # forces 
   wF = weights[:wF]
   F_dft = at.data[keys.F_key] * u"eV/Å"
   y_F = wF * reinterpret(eltype(F_dft[1]), F_dft)
   A_F = wF * reinterpret(eltype(efv.forces[1]), efv.forces)

   # # virial 
   # wV = weights[:wV]
   # V_dft = at.data[keys.V_key] * u"eV"
   # y_V = wV * V_dft[:]
   # # display( reinterpret(eltype(efv.virial), efv.virial) )
   # A_V = wV * reshape(reinterpret(eltype(efv.virial[1]), efv.virial), 9, :)

   return vcat(A_E, A_F), vcat(y_E, y_F)
end


function assemble_lsq(calc, data, weights, data_keys; 
                      rng = MersenneTwister(1234), 
                      executor = Folds.ThreadedEx())
   ps, st = Lux.setup(rng, calc)
   blocks = Folds.map(at -> local_lsqsys(calc, at, ps, st, 
                                         weights, data_keys), 
                      data, executor)
                         
   A = reduce(vcat, [b[1] for b in blocks])
   y = reduce(vcat, [b[2] for b in blocks])
   return A, y
end

function EF_err(sys::JuLIP.Atoms, calc)
   E = JuLIP.energy(calc, sys) 
   F = JuLIP.forces(calc, sys)
   E_ref = JuLIP.get_data(sys, "energy")
   F_ref = JuLIP.get_data(sys, "force")
   return abs(E - E_ref) / length(sys), norm.(F - F_ref)
end

function EF_err(sys::AtomsBase.AbstractSystem, calc)
   efv = M.energy_forces_virial(sys, calc)
   F_ustrip = [ ustrip.(f) for f in efv.forces ]
   E_ref = sys.data[:energy]
   F_ref = sys.data[:force]
   return abs(ustrip(efv.energy) - E_ref) / length(sys), norm.(F_ustrip - F_ref)
end
   
function rmse(test, calc) 
   E_errs = Float64[] 
   F_errs = Float64[] 
   for sys in test 
      E_err, F_err = EF_err(sys, calc)
      push!(E_errs, E_err)
      append!(F_errs, F_err) 
   end 
   return norm(E_errs) / sqrt(length(E_errs)), 
          norm(F_errs) / sqrt(length(F_errs))
end


function set_linear_params(calc, θ)
   # TODO: replace the first line with extracting the parameters 
   #       from the calculator!! 
   ps, st = Lux.setup(MersenneTwister(1234), calc.model)
   ps_lin = (WB = ps.WB, Wpair = ps.Wpair, pairbasis = NamedTuple(), rbasis = NamedTuple())
   _θ, restruct = destructure(ps_lin) 
   ps_lin_fit = restruct(θ)
   ps_fit = deepcopy(ps)
   ps_fit.WB[:] = ps_lin_fit.WB[:]
   ps_fit.Wpair[:] = ps_lin_fit.Wpair[:]
   calc_model_fit = M.ACEPotential(calc.model, ps_fit, st)
   return calc_model_fit
end

end 