

using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

using Optimisers, ForwardDiff, Unitful
import AtomsCalculators

using Random, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 15
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
                      level = level, max_level = max_level, maxl = 8, 
                      init_WB = :glorot_normal)

ps, st = LuxCore.setup(rng, model)

calc = M.ACEPotential(model, ps, st)

##

@info("Testing correctness of potential energy")
for ntest = 1:20 
   at = rattle!(bulk(:Si, cubic=true) * 2, 0.1)
   at_flex = AtomsBase.FlexibleSystem(at)
   nlist = JuLIP.neighbourlist(at, ustrip(M.cutoff_radius(calc)))
   E = 0.0 
   for i = 1:length(at)
      Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i)
      z0 = at.Z[i]
      E += M.evaluate(calc.model, Rs, Zs, z0, ps, st)[1]
   end
   efv = M.energy_forces_virial(at, calc, ps, st)
   E2 = AtomsCalculators.potential_energy(at_flex, calc)
   print_tf(@test abs(E - ustrip(efv.energy))/abs(E) < 1e-12)
   print_tf(@test abs(E - ustrip(E2)) / abs(E) < 1e-12)
end

##

@info("Testing correctness of forces ")
@info("   .... TODO TEST VIRIALS ..... ")

at = rattle!(bulk(:Si, cubic=true), 0.1)
at_flex = AtomsBase.FlexibleSystem(at)

@info(" consistency local vs EmpiricalPotentials implementation")
@info("this currently fails due to a bug in EmpiricalPotentials")
# efv1 = M.energy_forces_virial(at, calc, ps, st)
# efv2 = AtomsCalculators.energy_forces_virial(at_flex, calc)
# efv3 = M.energy_forces_virial_serial(at, calc, ps, st)
# print_tf(@test efv1.energy ≈ efv2.energy)
# print_tf(@test all(efv1.forces .≈ efv2.force))
# print_tf(@test efv1.virial ≈ efv1.virial)
# print_tf(@test efv1.energy ≈ efv3.energy)
# print_tf(@test all(efv1.forces .≈ efv3.forces))
# print_tf(@test efv1.virial ≈ efv3.virial)

##

@info("test consistency of forces with energy")
@info(" TODO: write virial test!")
for ntest = 1:10
   at = rattle!(bulk(:Si, cubic=true), 0.1)
   at.Z[[3,6,8]] .= 8
   Us = randn(SVector{3, Float64}, length(at)) / length(at)
   dF0 = - dot(Us, M.energy_forces_virial_serial(at, calc, ps, st).forces)
   X0 = deepcopy(at.X)
   F(t) = M.energy_forces_virial_serial(JuLIP.set_positions!(at, X0 + t * Us), 
                                    calc, ps, st).energy
   print_tf( @test ACEbase.Testing.fdtest(F, t -> dF0, 0.0; verbose=false ) )
end
println() 


##
# testing the AD through a loss function 
@info("Testing Zygote-AD through a loss function")

using Zygote 
using Unitful 
using Unitful: ustrip

# need to make sure that the weights in the loss remove the units! 
for (wE, wV, wF) in [ (1.0 / u"eV", 0.0 / u"eV", 0.0 / u"eV/Å"), 
                      (0.0 / u"eV", 1.0 / u"eV", 0.0 / u"eV/Å"), 
                      (0.0 / u"eV", 0.0 / u"eV", 1.0 / u"eV/Å"), 
                      (1.0 / u"eV", 0.1 / u"eV", 0.1 / u"eV/Å") ]
   # random structure 
   at = rattle!(bulk(:Si, cubic=true), 0.1)
   at.Z[[3,6,8]] .= 8

   # wE = 1.0 / u"eV"
   # wV = 1.0 / u"eV"
   # wF = 0.33 / u"eV/Å"

   function loss(at, calc, ps, st)
      efv = M.energy_forces_virial(at, calc, ps, st)
      _norm_sq(f) = sum(abs2, f)
      return (   wE^2 * efv.energy^2 / length(at)  
               + wV^2 * sum(abs2, efv.virial) / length(at) 
               + wF^2 * sum(_norm_sq, efv.forces) )
   end


   g = Zygote.gradient(ps -> loss(at, calc, ps, st), ps)[1] 

   p_vec, _restruct = destructure(ps)
   g_vec = destructure(g)[1]
   u = randn(length(p_vec)) / length(p_vec)
   dot(g_vec, u)
   _ps(t) = _restruct(p_vec + t * u)
   F(t) = loss(at, calc, _ps(t), st)
   dF0 = dot(g_vec, u)

   @info("(wE, wV, wF) = ($wE, $wV, $wF)")
   FDTEST = ACEbase.Testing.fdtest(F, t -> dF0, 0.0; verbose=true)
   println(@test FDTEST)
end