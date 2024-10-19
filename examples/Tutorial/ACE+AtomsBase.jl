# # ACEpotentials.jl + AtomsBase.jl Tutorial

# ## Introduction
# 
# This tutorial demonstrates how the ACEpotentials.jl package 
# interoperates with the AtomsBase.jl ecosystem. 

# ## Installation
#
# add and load general packages used in this notebook.

using Pkg
## Uncomment the next line if installing Julia for the first time
## Pkg.Registry.add("General")  
## Pkg.Registry.add
Pkg.add(["ExtXYZ", "Unitful", "Distributed", "AtomsCalculators",
         "Molly", "AtomsCalculatorsUtilities", "AtomsBuilder", 
         ])

## ACEpotentials installation:  
## If ACEpotentials has not been installed yet, uncomment the following lines
## Add the ACE registry, which stores the ACEpotential package information 
Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.add(["GeomOpt", "ACEpotentials",])

#   We can check the status of the installed packages.

using Pkg; Pkg.activate(".")
Pkg.status()

#   Import all the packages that we will be using, create some processes
#   for parallel model fitting 

using ExtXYZ, Unitful, AtomsCalculators, Distributed, ACEpotentials, 
      AtomsCalculatorsUtilities
using AtomsCalculatorsUtilities.SitePotentials: cutoff_radius
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

# ## Fit a potential for Cu 
#
# The tutorial can be adapted trivially to use datasets for Ni, Cu, Li, Mo, Si, Ge. 
#
# We generate a smallish model (about 300 basis functions) for Cu, using 
# correlation-order 3 (body-order 4), and default for rcut. Then we estimate 
# the model parameters using the standard BLR solver.

## generate a model for Cu
sym = :Cu 
model = ace1_model(elements = [sym,], order = 3, totaldegree = [ 20, 16, 12 ])
@show length_basis(model)
@show cutoff_radius(model)
## estimate parameters  
train, test, _ = ACEpotentials.example_dataset("Zuo20_$sym")
solver = ACEfit.BLR(; factorization = :svd)
acefit!(train, model;  solver=solver); GC.gc()
## quickly check test errors => 0.5 meV/atom and 0.014 eV/A are ok
ACEpotentials.compute_errors(test, model);

# ## Geometry Optimization with GeomOpt 
# 
# ( Note: we should use GeometryOptimization.jl, but this is not yet updated to 
#   AtomsBase.jl version 0.4. )
#
# First import some stuff + a little hack to make GeomOpt play nice with 
# the latest AtomsBase. This is a shortcoming of DecoratedParticles.jl 
# and requires some updates to fully implement the AtomsBase interface. 

using AtomsBuilder, GeomOpt, AtomsCalculators, AtomsBase
using AtomsBase: FlexibleSystem, FastSystem
using AtomsCalculators: potential_energy
function _flexiblesystem(sys) 
   c3ll = cell(sys)
   particles = [ AtomsBase.Atom(species(sys, i), position(sys, i)) 
                 for i = 1:length(sys) ] 
   return FlexibleSystem(particles, c3ll)
end; 

# We generate a cubic Cu unit cell, but our potential might not have the same 
# equilibrium bond distance as the default in AtomsBuilder, so we optimize 
# the unit cell. 

ucell = bulk(sym, cubic=true)
ucell, _ = GeomOpt.minimise(ucell, model; variablecell=true)

# We keep the energy of the equilibrated unit cell to later compute the 
# defect formation energy. 

Eperat = potential_energy(ucell, model) / length(ucell)
@show Eperat; 

# Now that we have an equilibrated unit cell we enlarge it, and then delete 
# an atom to generate a vacancy defect. 

sys = _flexiblesystem(ucell) * (2,2,2)
deleteat!(sys, 1)
sys 

# Now we do another geometry optimization to get the equilibrium geometry. 

vacancy_equil, result = GeomOpt.minimise(sys, model; variablecell = false)
@show result.g_residual;
                       
# We get an estimate of the formation energy. Note this is likely a poor 
# estimate since we didn't train the model on vacancy configurations. 

E_def = potential_energy(vacancy_equil, model) - length(sys) * Eperat
@show E_def;

# ## Molecular Dynamics with Molly
# 
# First import some stuff + a little hack to make GeomOpt play nice with 
# the latest AtomsBase. This is a shortcoming of DecoratedParticles.jl 
# and requires some updates to fully implement the AtomsBase interface. 

import Molly 
sys = rattle!(bulk(sym, cubic=true) * (2,2,2), 0.03)
sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
temp = 298.0u"K"
sys_md = Molly.System(
   sys_md;
   general_inters = (model,),
   velocities = Molly.random_velocities(sys_md, temp),
   loggers=(temp=Molly.TemperatureLogger(100),) )
## energy = Molly.PotentialEnergyLogger(100),), )
## can't add an energy logger because Molly internal energies are per mol
simulator = Molly.VelocityVerlet(
   dt = 1.0u"fs",
   coupling = Molly.AndersenThermostat(temp, 1.0u"ps"), )

Molly.simulate!(sys_md, simulator, 1000)

## the temperature seems to fluctuate a bit, but at least it looks stable?
@info("Temperature history:", sys_md.loggers.temp.history)

