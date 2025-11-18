
# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, Test, LinearAlgebra, ACEbase 
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

# build a pure Lux Rnl basis compatible with LearnableRnlrzz
import EquivariantTensors as ET
using StaticArrays
using Lux

# In ET we currently store an edge xij as a NamedTuple, e.g, 
#    xij = (𝐫ij = ..., zi = ..., zj = ...)
# The NTtransform is a wrapper for mapping xij -> y 
# (in this case y = transformed distance) adding logic to enable 
# differentiation through this operation. 
#
et_trans = let _i2z = basis._i2z, transforms = basis.transforms
   ET.NTtransform(x -> begin
         idx_i = M._z2i(basis, x.zi)
         idx_j = M._z2i(basis, x.zj)
         trans_ij = basis.transforms[idx_i, idx_j] 
         r = norm(x.𝐫ij)
         return trans_ij(r)
      end)
   end 

# the envelope is always a simple quartic (1 -x^2)^2
#  (note the transforms is normalized to map to [-1, 1])
et_env = y -> (1 - y^2)^2

# the polynomial basis 
et_polys = basis.polys

# the linear layer transformation 
# selector maps a (Zi, Zj) pair to an index a for transforming 
#   P(yij) -> W[a] * P(zij) 
# with W[a] learnable weights 
selector = let _i2z = basis._i2z
   x -> begin 
         iz = M._z2i(basis, x.zi)
         jz = M._z2i(basis, x.zj)
         return (iz - 1) * length(_i2z) + jz
      end 
   end 
#                           indim           outdim         4 categories
et_linl = ET.SelectLinL(length(et_polys), size(ps.Wnlq, 1), 4, selector)

et_rbasis = SkipConnection(   # input is xij 
         Chain(y = et_trans,  # transforms yij 
               P = SkipConnection(
                     et_polys, 
                     WrappedFunction( Py -> et_env.(Py[2]) .* Py[1] )
                  )
               ),   # r -> y -> P = e(y) * polys(y)
         et_linl    # P -> W(Zi, Zj) * P 
      )
et_ps, et_st = Lux.setup(Random.default_rng(), et_rbasis)

# translate the weights from the AP basis to the ET basis 
et_ps.connection.W[:, :, 1] = ps.Wnlq[:, :, 1, 1]
et_ps.connection.W[:, :, 2] = ps.Wnlq[:, :, 1, 2]
et_ps.connection.W[:, :, 3] = ps.Wnlq[:, :, 2, 1]
et_ps.connection.W[:, :, 4] = ps.Wnlq[:, :, 2, 2]

for ntest = 1:100 
   r = 2 + rand() 
   Zi = rand(basis._i2z)
   Zj = rand(basis._i2z)
   xij = ( 𝐫ij = SA[r,0,0], zi = Zi, zj = Zj)

   P_ap = basis(r, Zi, Zj, ps, st)
   P_et, _ = et_rbasis(xij, et_ps, et_st)
   print_tf(@test P_ap ≈ P_et) 
end