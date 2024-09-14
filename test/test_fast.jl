
# using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ))
##

using ACEpotentials
using LazyArtifacts, ExtXYZ, LinearAlgebra, Test
using Polynomials4ML.Testing: print_tf
using AtomsCalculators: energy_forces_virial, potential_energy

M = ACEpotentials.Models
## ----- setup -----

@info("-------- Test Fast ACE evaluator ----------")

@info("construct a Si model and fit parameters using BLR")
model = ace1_model(elements = [:Si],
                   Eref = [:Si => -158.54496821],
                   rcut = 5.5,
                   order = 3,
                   totaldegree = 10)
data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

acefit!(data, model;
      data_keys..., weights = weights, 
      solver = ACEfit.BLR())

# artificially make some parameters zero 
Ism = findall(abs.(model.ps.WB) .< 1e-2)
model.ps.WB[Ism] .= 0.0

# construct the fast evaluator (is it actually fast??)
fpot = M.fast_evaluator(model)
fpot_d = M.fast_evaluator(model, aa_static=false)

## 

using StaticArrays, AtomsBase

for ntest = 1:20 
   Rs, Zs, z0 = M.rand_atenv(model.model, rand(8:12))

   E1 = M.eval_site(fpot, Rs, Zs, z0)
   E2 = M.eval_site(model, Rs, Zs, z0)
   v1, ∇v1 = M.eval_grad_site(fpot, Rs, Zs, z0)
   v2, ∇v2 = M.eval_grad_site(model, Rs, Zs, z0)
   v3, ∇v3 = M.eval_grad_site(fpot_d, Rs, Zs, z0)

   print_tf(@test E1 ≈ E2 ≈ v1 ≈ v2 ≈ v3)
   print_tf(@test all(∇v1 .≈ ∇v2 .≈ ∇v3))
end 
println()

## 

# using BenchmarkTools
# @btime M.eval_site($fpot, $Rs, $Zs, $z0)
# @btime M.eval_site($model, $Rs, $Zs, $z0)

##

@info("confirm that predictions are identical")

using AtomsBuilder, AtomsCalculators, Unitful
using AtomsCalculators: potential_energy
tolerance = 1e-10 
rattle = 0.1 

for ntest = 1:20
   local at, efv1, efv2, efv3  
   at = bulk(:Si, cubic=true) * 2 
   rattle!(at, rattle)
   efv1 = energy_forces_virial(at, model) 
   efv2 = energy_forces_virial(at, fpot)
   efv3 = energy_forces_virial(at, fpot_d)
   print_tf(@test ustrip(abs(efv1.energy - efv2.energy)) < tolerance)
   print_tf(@test all(efv1.forces .≈ efv2.forces .≈ efv3.forces))
   print_tf(@test all(efv1.virial .≈ efv2.virial .≈ efv3.virial))
end
println() 

##

@info("construct a TiAl model and set random parameters")

model = ace1_model(elements = [:Ti, :Al],
					    order = 3,
					    totaldegree = 8, 
					    rcut = 5.5, 
					    Eref = [:Ti => -1.586, :Al => -1.055])  # BAD E0s, don't copy!!!

model.ps.WB .= randn( size(model.ps.WB) )
lenB = size(model.ps.WB, 1)
model.ps.WB[:, :] = Diagonal((1:lenB).^(-2)) * model.ps.WB[:, :]
model.ps.Wpair[:] = randn(size(model.ps.Wpair))
len2 = size(model.ps.Wpair, 1)
model.ps.Wpair[:, :] = Diagonal((1:len2).^(-2)) * model.ps.Wpair[:, :]

##

@info("convert to UF_ACE format")      
fpot = M.fast_evaluator(model)
fpot_d = M.fast_evaluator(model, aa_static=false)

##

@info("confirm that predictions are identical on a site")

for ntest = 1:20 
   Rs, Zs, z0 = M.rand_atenv(model.model, rand(8:12))
   E1 = M.eval_site(fpot, Rs, Zs, z0)
   E2 = M.eval_site(model, Rs, Zs, z0)
   v1, ∇v1 = M.eval_grad_site(fpot, Rs, Zs, z0)
   v2, ∇v2 = M.eval_grad_site(model, Rs, Zs, z0)
   v3, ∇v3 = M.eval_grad_site(fpot_d, Rs, Zs, z0)

   print_tf(@test E1 ≈ E2 ≈ v1 ≈ v2 ≈ v3)
   print_tf(@test all(∇v1 .≈ ∇v2 .≈ ∇v3))

end 
println() 

##

@info("confirm that predictions are identical on systems")

tolerance = 1e-12 
rattle = 0.01 

for ntest = 1:20
   local sys, efv1, efv2, efv3 
   sys = rattle!(bulk(:Al, cubic=true) * 2, 0.1)
   randz!(sys, [:Ti => 0.5, :Al => 0.5])

   efv1 = energy_forces_virial(sys, model) 
   efv2 = energy_forces_virial(sys, fpot)
   efv3 = energy_forces_virial(sys, fpot_d)
   print_tf(@test ustrip(abs(efv1.energy - efv2.energy)) < tolerance)
   print_tf(@test all(efv1.forces .≈ efv2.forces .≈ efv3.forces))
   print_tf(@test all(efv1.virial .≈ efv2.virial .≈ efv3.virial))
end
println() 
