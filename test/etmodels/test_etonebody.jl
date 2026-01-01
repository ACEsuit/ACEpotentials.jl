# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "EquivariantTensors.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "Polynomials4ML.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "DecoratedParticles"))

##

using ACEpotentials, StaticArrays, Lux, AtomsBase, AtomsBuilder, Unitful, 
      AtomsCalculators, Random, LuxCore, Test, LinearAlgebra, ACEbase 

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels
import EquivariantTensors as ET
import Polynomials4ML as P4ML 
import DecoratedParticles as DP

using Polynomials4ML.Testing: print_tf, println_slim

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##

# Generate an ACE model in the v0.8 style but 
#  - with fixed rcut. (relaxes this requirement later!!)
#  - remove E0s
#  - remove pair potential 

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 10
order = 3 
maxl = 6

# modify rin0cuts to have same cutoff for all elements 
# TODO: there is currently a bug with variable cutoffs 
#       (?is there? The radials seem fine? check again)
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

E0s_ref = Dict(:Si => randn(), :O => randn())

model = M.ace_model(; elements = elements, order = order, 
            Ytype = :solid, level = level, max_level = max_level, 
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,  
            init_WB = :glorot_normal, init_Wpair = :glorot_normal, 
            E0s = E0s_ref
            )

ps, st = Lux.setup(rng, model)          

##

V0 = model.Vref
E0s_z = V0.E0   # Dict{Int, Float64} 
et_E0s = Dict( ChemicalSpecies(key) => val for (key, val) in E0s_z )

# let block is only needed to avoid type instability 
catfun = let  
   x -> x.z 
end
et_V0 = ETM.one_body(et_E0s, catfun)
ps, st = Lux.setup(rng, et_V0)

## 



function rand_struct() 
   sys = AtomsBuilder.bulk(:Si) * (2,2,2)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 

function site_Es_old(V0, sys) 
   return [ M.eval_site(V0, [], [], atomic_number(sys, i)) 
            for i in 1:length(sys) ]
end
   
function site_Es_et(et_V0, sys, args...) 
   G = ET.Atoms.interaction_graph(sys, 5.0 * u"Å")
   return et_V0(G, args...) 
end

##

@info("Confirm correctness of site energies")

for ntest = 1:30 
   sys = rand_struct()
   Es = site_Es_old(V0, sys)
   et_Es_a = site_Es_et(et_V0, sys)
   et_Es_b, _ = site_Es_et(et_V0, sys, ps, st)
   print_tf( @test Es ≈ et_Es_a ≈ et_Es_b )
end
println() 

##

@info("Confirm correctness of gradient")

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, 5.0 * u"Å")
∂G1 = ETM.site_grads(et_V0, G, ps, st)

# ETOneBody returns NamedTuple with edge_data filled with empty VState() elements
# Empty VState() acts as additive identity: VState(r=...) + VState() == VState(r=...)
println_slim(@test ∂G1 isa NamedTuple)
println_slim(@test haskey(∂G1, :edge_data))
println_slim(@test length(∂G1.edge_data) == length(G.edge_data))
println_slim(@test all(v -> v == DP.VState(), ∂G1.edge_data))

##

@info("Confirm correctness of basis and basis jacobian") 

𝔹1 = ETM.site_basis(et_V0, G, ps, st) 
𝔹2, ∂𝔹2 = ETM.site_basis_jacobian(et_V0, G, ps, st) 

println_slim(@test size(𝔹1) == size(𝔹2) == (length(sys), 0))
println_slim(@test size(∂𝔹2) == (ET.maxneigs(G), length(sys), 0))

##

# turn off during CI -- need to sort out CI for GPU tests 
#=
@info("Check GPU evaluation") 
using Metal 
dev = Metal.mtl
ps_32 = ET.float32(ps)
st_32 = ET.float32(st)
ps_dev = dev(ps_32)
st_dev = dev(st_32)

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, 5.0 * u"Å")
G_32 = ET.float32(G)
G_dev = dev(G_32)

E1, st = et_V0(G_32, ps_32, st_32)
E2_dev, st_dev = et_V0(G_dev, ps_dev, st_dev)
E2 = Array(E2_dev)
# TODO: add E1 ≈ E2 test??

g1 = ETM.site_grads(et_V0, G_32, ps_32, st_32)
g2_dev = ETM.site_grads(et_V0, G_dev, ps_dev, st_dev)
g2 = Array(g2_dev)
println_slim(@test g1 == g2)

b1 = ETM.site_basis(et_V0, G_32, ps_32, st_32)
b2_dev = ETM.site_basis(et_V0, G_dev, ps_dev, st_dev)
b2 = Array(b2_dev)
println_slim(@test b1 == b2)

b1, ∂db1 = ETM.site_basis_jacobian(et_V0, G_32, ps_32, st_32)
b2_dev, ∂db2_dev = ETM.site_basis_jacobian(et_V0, G_dev, ps_dev, st_dev)
b2 = Array(b2_dev)
∂db2 = Array(∂db2_dev)
println_slim(@test b1 == b2)
println_slim(@test ∂db1 == ∂db2)
=#