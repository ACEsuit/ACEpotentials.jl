

# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

##

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

using Optimisers, ForwardDiff, Unitful
import AtomsCalculators

using AtomsBuilder, EmpiricalPotentials
AB = AtomsBuilder
using EmpiricalPotentials: get_neighbours


using Random, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 15
order = 3
E0s = Dict( :Si => -158.54496821u"eV", 
            :O => -2042.0330099956639u"eV")
NZ = length(elements)

function make_rin0cut(zi, zj) 
   r0 = ACE1x.get_r0(zi, zj)
   return (rin = 0.0, r0 = r0, rcut = 6.5)
end

rin0cuts = SMatrix{NZ, NZ}([make_rin0cut(zi, zj) for zi in elements, zj in elements])


model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
                      level = level, max_level = max_level, maxl = 8, 
                      pair_maxn = 15, 
                      init_WB = :glorot_normal, init_Wpair = :glorot_normal,
                      E0s = E0s,
                      rin0cuts = rin0cuts
                      )

ps, st = LuxCore.setup(rng, model)

calc = M.ACEPotential(model, ps, st)   

##

@info("Testing correctness of E0s")
ps_vec, _restruct = destructure(ps)
ps_zero = _restruct(zero(ps_vec))

for ntest = 1:20
   local Rs, Zs, z0, at, efv 
   at = AB.randz!( AB.rattle!(AB.bulk(:Si, cubic=true) * 2, 0.1), 
                   (:Si => 0.6, :O => 0.5), )
   n_Si = count(x -> x == 14, AtomsBase.atomic_number(at))                   
   n_O = count(x -> x == 8, AtomsBase.atomic_number(at))
   nlist = PairList(at, M.cutoff_radius(calc))
   efv = M.energy_forces_virial(at, calc, ps_zero, st)
   print_tf(@test ustrip(abs(efv.energy - E0s[:Si] * n_Si - E0s[:O] * n_O)) < 1e-10)
end

println()

##

@info("Testing correctness of potential energy")
for ntest = 1:20 
   local Rs, Zs, z0, at, efv 

   at = AB.rattle!(AB.bulk(:Si, cubic=true) * 2, 0.1)
   nlist = PairList(at, M.cutoff_radius(calc))
   E = 0.0 
   for i = 1:length(at)
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      E += M.evaluate(calc.model, Rs, Zs, z0, ps, st)[1]
   end
   efv = M.energy_forces_virial(at, calc, ps, st)
   E2 = AtomsCalculators.potential_energy(at, calc)
   print_tf(@test abs(E - ustrip(efv.energy))/abs(E) < 1e-12)
   print_tf(@test abs(E - ustrip(E2)) / abs(E) < 1e-12)
end
println() 


##

@info("Testing correctness of forces ")
@info("   .... TODO TEST VIRIALS ..... ")

at = AB.rattle!(AB.bulk(:Si, cubic=true), 0.1)

@info(" consistency local vs EmpiricalPotentials implementation")
efv1 = M.energy_forces_virial(at, calc, ps, st)
efv2 = AtomsCalculators.energy_forces_virial(at, calc)
efv3 = M.energy_forces_virial_serial(at, calc, ps, st)
print_tf(@test efv1.energy ≈ efv2.energy)
print_tf(@test all(efv1.forces .≈ efv2.forces))
print_tf(@test efv1.virial ≈ efv1.virial)
print_tf(@test efv1.energy ≈ efv3.energy)
print_tf(@test all(efv1.forces .≈ efv3.forces))
print_tf(@test efv1.virial ≈ efv3.virial)

##

@info("test consistency of forces with energy")
@info(" TODO: write virial test!")
for ntest = 1:10
   local at, Us, dF0, X0, F, Z

   at = AB.rattle!(AB.bulk(:Si, cubic=true), 0.1)
   Z = AtomsBase.atomic_number(at)
   Z[[3,6,8]] .= 8
   at = AtomsBuilder.set_elements(at, Z)
   Us = randn(SVector{3, Float64}, length(at)) / length(at) * u"Å"
   dF0 = - dot(Us, M.energy_forces_virial_serial(at, calc, ps, st).forces)
   X0 = AtomsBase.position(at)
   F(t) = M.energy_forces_virial_serial(
               AtomsBuilder.set_positions(at, X0 + t * Us), 
               calc, ps, st).energy |> ustrip 
   print_tf( @test ACEbase.Testing.fdtest(F, t -> ustrip(dF0), 0.0; verbose=false ) )
end
println()

##

@info("check splinification of calculator")

lin_calc = M.splinify(calc, ps)
ps_lin, st_lin = LuxCore.setup(rng, lin_calc)
ps_lin.WB[:] .= ps.WB[:] 
ps_lin.Wpair[:] .= ps.Wpair[:]

for ntest = 1:10
   len = 10 
   mae = sum(1:len) do _
      at = AB.rattle!(AB.bulk(:Si, cubic=true), 0.1)
      Z = AtomsBase.atomic_number(at)
      Z[[3,6,8]] .= 8
      E = M.energy_forces_virial(at, calc, ps, st).energy
      E_lin = M.energy_forces_virial(at, lin_calc, ps_lin, st_lin).energy
      abs(E - E_lin) / (abs(E) + abs(E_lin))
   end
   mae /= len 
   print_tf(@test mae < 1e-3)
end
println() 

##

@info("Test splinified calculator basis usage")

for ntest = 1:10
   local ps_lin, st_lin, at, efv, _restruct

   ps_lin, st_lin = LuxCore.setup(rng, lin_calc)
   at = AB.rattle!(AB.bulk(:Si, cubic=true), 0.1)
   Z = AtomsBase.atomic_number(at)
   Z[[3,6,8]] .= 8

   efv = M.energy_forces_virial(at, lin_calc, ps_lin, st_lin)
   efv_b = M.energy_forces_virial_basis(at, lin_calc, ps_lin, st_lin)

   ps_vec, _restruct = destructure(ps_lin)
   print_tf(@test dot(efv_b.energy, ps_vec) ≈ efv.energy )
   print_tf(@test all(efv_b.forces * ps_vec .≈ efv.forces) )
   print_tf(@test sum(ps_vec .* efv_b.virial) ≈ efv.virial )
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
   local at, Z, dF0, g, _restruct, g_vec 

   # random structure 
   at = AB.rattle!(AB.bulk(:Si, cubic=true), 0.1)
   Z = AtomsBase.atomic_number(at)
   Z[[3,6,8]] .= 8

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
   FDTEST = ACEbase.Testing.fdtest(F, t -> dF0, 0.0; verbose=false)
   println(@test FDTEST)
end

##
