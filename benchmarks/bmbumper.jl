

using ACEpotentials, StrideArrays, Bumper, BenchmarkTools


## 

M = ACEpotentials.Models

using Random, LuxCore, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: print_tf
rng = Random.MersenneTwister(1234)

##

max_level = 8; maxl = 3; maxn = max_level; 
basis = M.ace_learnable_Rnlrzz(; level=M.TotalDegree(), max_level=max_level, 
                                 maxl = maxl, maxn = maxn, elements = (:Si, :O))
ps, st = LuxCore.setup(rng, basis)


##

Nat = 12
Rs, Zs, Z0 = M.rand_atenv(basis, Nat)
rs = norm.(Rs)

@info("Original implementation")
@btime M.evaluate_batched($basis, $rs, $Z0, $Zs, $ps, $st)

@info("In-place implementation (baby-bumper)")
Rnl = zeros(M.whatalloc(basis, rs, Z0, Zs, ps, st)...)
@btime M.evaluate_batched!(Rnl, $basis, $rs, $Z0, $Zs, $ps, $st)


## 

