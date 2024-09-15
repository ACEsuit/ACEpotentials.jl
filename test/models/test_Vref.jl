
# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
##

using Test
using Polynomials4ML.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

using Random, StaticArrays, LinearAlgebra, Unitful, EmpiricalPotentials
using AtomsCalculatorsUtilities.SitePotentials: eval_site, eval_grad_site
using AtomsBase
##

elements = [:C, :O, :H]
E0s = Dict(:C => -rand(), :O => -rand(), :H => -rand())
rcut = 5.5u"Å"
zbl = ZBL(rcut)
V0 = M.OneBody(E0s)

Vref1 = M._make_Vref(elements, E0s, false)
Vref2 = M._make_Vref(elements, nothing, true, ustrip(rcut))
Vref3 = M._make_Vref(elements, E0s, true, ustrip(rcut))

zC = atomic_number(ChemicalSpecies(:C))
zO = atomic_number(ChemicalSpecies(:O))
zH = atomic_number(ChemicalSpecies(:H))

Rs0 = SVector{3, Float64}[]
Zs0 = Int[]
nZ = rand(3:5)
Rs1 = randn(SVector{3, Float64}, nZ)
Zs1 = rand([zC, zO, zH], nZ)

##

print_tf(@test (eval_site(zbl, Rs0, Zs0, zC) == 0.0 ))
print_tf(@test (eval_site(zbl, Rs0, Zs0, zO) == 0.0 ))
print_tf(@test (eval_site(zbl, Rs0, Zs0, zH) == 0.0 ))
print_tf(@test (eval_site(V0, Rs0, Zs0, zC) == E0s[:C] ))
print_tf(@test (eval_site(V0, Rs0, Zs0, zO) == E0s[:O] ))
print_tf(@test (eval_site(V0, Rs0, Zs0, zH) == E0s[:H] ))
println() 

##

for (Rs, Zs) in [ (Rs0, Zs0), (Rs1, Zs1)], z in [zC, zO, zH]
   print_tf(@test ( eval_site(Vref1, Rs, Zs, z) == eval_site(V0, Rs, Zs, z) ))
   print_tf(@test ( eval_site(Vref2, Rs, Zs, z) == eval_site(zbl, Rs, Zs, z) ))
   print_tf(@test ( eval_site(Vref3, Rs, Zs, z) == eval_site(V0, Rs, Zs, z) + eval_site(zbl, Rs, Zs, z) ))
   print_tf(@test ( eval_grad_site(Vref1, Rs, Zs, z) == eval_grad_site(V0, Rs, Zs, z) ))
   print_tf(@test ( eval_grad_site(Vref2, Rs, Zs, z) == eval_grad_site(zbl, Rs, Zs, z) ))
   print_tf(@test ( eval_grad_site(Vref3, Rs, Zs, z)[2] == eval_grad_site(V0, Rs, Zs, z)[2] + eval_grad_site(zbl, Rs, Zs, z)[2] ))
end
println() 

##
