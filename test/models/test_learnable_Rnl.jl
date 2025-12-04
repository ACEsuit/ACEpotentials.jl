
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
using TestEnv; TestEnv.activate();
Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "EquivariantTensors.jl"))
Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "Polynomials4ML.jl"))

##

using ACEpotentials
M = ACEpotentials.Models
import EquivariantTensors as ET

using Random, LuxCore, Test, LinearAlgebra, ACEbase
using AtomsBase, StaticArrays
using Polynomials4ML.Testing: print_tf, println_slim
rng = Random.MersenneTwister(1234)

Random.seed!(1234)

##

max_level = 8 
level = M.TotalDegree()
maxl = 3; maxn = max_level; 
elements = (:Si, :O)
basis = M.ace_learnable_Rnlrzz(; level=level, max_level=max_level, 
                                 maxl = maxl, maxn = maxn, elements = elements)

ps, st = LuxCore.setup(rng, basis)

r = 3.0 
Zi = basis._i2z[1]
Zj = basis._i2z[2]
Rnl, st1 = basis(r, Zi, Zj, ps, st)
Rnl, Rnl_d = M.evaluate_ed(basis, r, Zi, Zj, ps, st)

@info("Test derivatives of Rnlrzz basis")

for ntest = 1:20 
   global ps, st
   r = 2.0 + rand() 
   Zi = rand(basis._i2z)
   Zj = rand(basis._i2z)
   U = randn(eltype(Rnl), length(Rnl))
   F(t) = dot(U, basis(r + t, Zi, Zj, ps, st))
   dF(t) = dot(U, M.evaluate_ed(basis, r + t, Zi, Zj, ps, st)[2])
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

@info("LearnableRnlrzz : Consistency of single and batched evaluation")

for ntest = 1:20 
   global ps, st
   Nat = rand(8:16)
   Rs, Zs, Z0 = M.rand_atenv(basis, Nat)
   rs = norm.(Rs)

   Rnl = [ M.evaluate(basis, r, Z0, z, ps, st) for (r, z) in zip(rs, Zs) ]
   Rnl_b = M.evaluate_batched(basis, rs, Z0, Zs, ps, st)
   print_tf(@test all([Rnl_b[j, :] for j = 1:Nat] .≈ Rnl))

   Rnl_b2, ∇Rnl_b = M.evaluate_ed_batched(basis, rs, Z0, Zs, ps, st)
   ∇Rnl = [ M.evaluate_ed(basis, r, Z0, z, ps, st)[2]
            for (r, z) in zip(rs, Zs) ] 
                   
   print_tf(@test Rnl_b ≈ Rnl_b2)
   print_tf(@test all(∇Rnl .≈ [∇Rnl_b[j, :] for j = 1:Nat ]))
end

## 

@info("quick splinification check")
basis_spl = M.splinify(basis, ps; nnodes = 100)
ps_spl, st_spl = LuxCore.setup(rng, basis_spl)

Rnl = basis(r, Zi, Zj, ps, st)
Rnl_spl = basis_spl(r, Zi, Zj, ps_spl, st_spl)

println_slim(@test norm(Rnl - Rnl_spl, Inf) < 1e-4)

Rnl, ∇Rnl = M.evaluate_ed(basis, r, Zi, Zj, ps, st)
Rnl_spl, ∇Rnl_spl = M.evaluate_ed(basis_spl, r, Zi, Zj, ps_spl, st_spl)

println_slim(@test norm(Rnl - Rnl_spl, Inf) < 1e-4 )
println_slim(@test norm(∇Rnl - ∇Rnl_spl, Inf) < 1e-2 )

##
#
# test the conversion to a Lux style Rnl basis 
# 
et_rbasis = M._convert_Rnl_learnable(basis)
et_ps, et_st = LuxCore.setup(Random.default_rng(), et_rbasis)

et_ps.connection.W[:, :, 1] = ps.Wnlq[:, :, 1, 1]
et_ps.connection.W[:, :, 2] = ps.Wnlq[:, :, 1, 2]
et_ps.connection.W[:, :, 3] = ps.Wnlq[:, :, 2, 1]
et_ps.connection.W[:, :, 4] = ps.Wnlq[:, :, 2, 2]

for ntest = 1:50 
   global ps, st, et_ps, et_st 
   r = 2.0 + 5 * rand()
   Zi = rand(basis._i2z)
   Zj = rand(basis._i2z)
   xij = ( 𝐫 = SA[r,0.0,0.0], s0 = ChemicalSpecies(Zi), s1 = ChemicalSpecies(Zj) )
   R1 = basis(r, Zi, Zj, ps, st)
   R2 = et_rbasis( xij, et_ps, et_st)[1] 
   print_tf(@test R1 ≈ R2)
end 

# batched test 
for ntest = 1:10 
   z0 = rand(basis._i2z)
   xx = [ (𝐫 = SA[2.0 + 2 * rand(), 0.0, 0.0], 
          s0 = ChemicalSpecies(z0), 
          s1 = ChemicalSpecies(rand(basis._i2z))) for _ in 1:30 ]
   rr = [ x.𝐫[1] for x in xx ]
   Zjs = [ atomic_number(x.s1) for x in xx ]
   R1 = M.evaluate_batched(basis, rr, z0, Zjs, ps, st)
   R2 = et_rbasis( xx, et_ps, et_st)[1]
   print_tf(@test R1 ≈ R2) 
end

## 
# run on GPU 
using Metal 
dev = Metal.mtl 

z0 = rand(basis._i2z)
xx = [ (𝐫 = SA[2.0 + 2 * rand(), 0.0, 0.0], 
        s0 = ChemicalSpecies(z0), 
        s1 = ChemicalSpecies(rand(basis._i2z))) for _ in 1:1000 ]

xx_dev = dev(ET.float32.(xx))
ps_dev = dev(ET.float32(et_ps))
st_dev = dev(ET.float32(et_st))

R1 = et_rbasis(xx, et_ps, et_st)[1]
R2 = et_rbasis(xx_dev, ps_dev, st_dev)

# this has a scalar indexing error, whereas the following tests work ok 

## 

trans1 = ET.NTtransform( x -> norm(x.𝐫) )
ps, st = LuxCore.setup(rng, trans1)
y1 = trans1(xx, ps, st)[1] 
y1_dev = trans1(xx_dev, ps, st)[1] 

trans2 = et_rbasis.layers.layers.y
ps2, st2 = LuxCore.setup(rng, trans2)
y2 = trans2(xx, ps2, st2)[1]
st2_dev = dev(ET.float32(st2))
y2_dev = trans2(xx_dev, ps2, st2_dev)[1]

## 
using Lux, Polynomials4ML
import Polynomials4ML as P4ML

l_P = et_rbasis.layers.layers.P.layers
l_yP = Chain(; 
      y = trans2, 
      P = l_P ) 
ps, st = LuxCore.setup(rng, l_yP)
st_dev = dev(ET.float32(st))
P1 = l_yP(xx, ps, st)[1]
P1_dev = l_yP(xx_dev, ps, st_dev)[1]
