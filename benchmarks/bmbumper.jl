

using ACEpotentials, StrideArrays, Bumper, BenchmarkTools, SpheriCart


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
@btime M.evaluate_batched!($Rnl, $basis, $rs, $Z0, $Zs, $ps, $st)

Rnl1, _ = M.evaluate_batched(basis, rs, Z0, Zs, ps, st)
M.evaluate_batched!(Rnl, basis, rs, Z0, Zs, ps, st)
Rnl1 ≈ Rnl

##

# same with splines?
bspl = M.splinify(basis, ps)
ps, st = LuxCore.setup(rng, bspl)

@info("Original implementation")
@btime M.evaluate_batched($bspl, $rs, $Z0, $Zs, $ps, $st)

@info("In-place implementation (baby-bumper)")
Rnl = zeros(M.whatalloc(bspl, rs, Z0, Zs, ps, st)...)
@btime M.evaluate_batched!($Rnl, $bspl, $rs, $Z0, $Zs, $ps, $st)

Rnl1, _ = M.evaluate_batched(bspl, rs, Z0, Zs, ps, st)
M.evaluate_batched!(Rnl, bspl, rs, Z0, Zs, ps, st)
Rnl1 ≈ Rnl

## 
# next step : inner kernel 

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 15
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
                      level = level, max_level = max_level, maxl = 8, 
                      pair_maxn = 15, 
                      init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = LuxCore.setup(rng, model)

##

Nat = rand(8:16)
Rs, Zs, Z0 = M.rand_atenv(model, Nat)
rs = norm.(Rs)

Rnl, _ = M.evaluate_batched(model.rbasis, rs, Z0, Zs, ps.rbasis, st.rbasis)
Ylm = SpheriCart.compute(model.ybasis, Rs)

B, _ = M.evaluate(model.tensor, Rnl, Ylm);
@btime M.evaluate($(model.tensor), $Rnl, $Ylm);

alcB, alc_interm = M.whatalloc(model.tensor, Rnl, Ylm)
B1 = zeros(alcB...)
intm = (_AA = zeros(alc_interm._AA...), )
@btime M.evaluate!($B1, $(model.tensor), $Rnl, $Ylm, $intm)

B, _ = M.evaluate(model.tensor, Rnl, Ylm);
M.evaluate!(B1, model.tensor, Rnl, Ylm, intm)
B ≈ B1

## 

val1, _ = M.evaluate(model, Rs, Zs, Z0, ps, st)
val2, _ = M.evaluate_bump(model, Rs, Zs, Z0, ps, st)
val1 ≈ val2

@info("old evaluate")
@btime M.evaluate($model, $Rs, $Zs, $Z0, $ps, $st)

@info("with bumper")
@btime M.evaluate_bump($model, $Rs, $Zs, $Z0, $ps, $st)

##

@code_warntype M.evaluate_bump(model, Rs, Zs, Z0, ps, st)

@profview let model = model, Rs = Rs, Zs = Zs, Z0 = Z0, ps = ps, st = st
   v = 0.0 
   for nrun = 1:200_000
      v += M.evaluate_bump(model, Rs, Zs, Z0, ps, st)[1]
   end
end
