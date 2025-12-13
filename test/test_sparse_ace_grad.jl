using ACEpotentials
M = ACEpotentials.Models

import EquivariantTensors as ET
import Polynomials4ML as P4ML

using StaticArrays, Lux
using AtomsBase, AtomsBuilder, Unitful, AtomsCalculators

using Random, LuxCore, Test, LinearAlgebra
using Zygote

rng = Random.MersenneTwister(1234)

# Minimal setup
elements = (:Si,)
level = M.TotalDegree()
max_level = 4
order = 2
maxl = 1

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

model = M.ace_model(; elements = elements, order = order,
            Ytype = :solid, level = level, max_level = max_level,
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, model)

# Build the tensor
rbasis = model.rbasis
et_i2z = AtomsBase.ChemicalSpecies.(rbasis._i2z)
et_rbasis = M._convert_Rnl_learnable(rbasis; zlist = et_i2z, rfun = x -> norm(x.𝐫))
et_rspec = rbasis.spec

et_ybasis = Chain(𝐫ij = ET.NTtransform(x -> x.𝐫), Y = model.ybasis)
et_yspec = P4ML.natural_indices(et_ybasis.layers.Y)

AA_spec = model.tensor.meta["𝔸spec"]
et_mb_spec = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA_spec])

et_mb_basis = ET.sparse_equivariant_tensor(
      L = 0,
      mb_spec = et_mb_spec,
      Rnl_spec = et_rspec,
      Ylm_spec = et_yspec,
      basis = real
   )

# Test directly calling evaluate on the tensor
Rnl_test = rand(10)
Ylm_test = rand(3)

et_mb_ps, et_mb_st = LuxCore.setup(rng, et_mb_basis)

# Test 1: Gradient w.r.t. Rnl directly through evaluate
println("Test 1: Gradient through evaluate w.r.t Rnl...")
try
    g = Zygote.gradient(Rnl -> sum(ET.evaluate(et_mb_basis, Rnl, Ylm_test, et_mb_ps, et_mb_st)[1]), Rnl_test)[1]
    println("  Success, Gradient shape: ", size(g))
catch e
    println("  Failed: ", e)
    showerror(stdout, e, catch_backtrace())
end

# Test 2: Gradient w.r.t. Ylm
println("\nTest 2: Gradient through evaluate w.r.t Ylm...")
try
    g = Zygote.gradient(Ylm -> sum(ET.evaluate(et_mb_basis, Rnl_test, Ylm, et_mb_ps, et_mb_st)[1]), Ylm_test)[1]
    println("  Success, Gradient shape: ", size(g))
catch e
    println("  Failed: ", e)
end

# Test 3: Gradient through the functor call
println("\nTest 3: Gradient through functor call...")
try
    g = Zygote.gradient(Rnl -> sum(et_mb_basis((Rnl, Ylm_test), et_mb_ps, et_mb_st)[1][1]), Rnl_test)[1]
    println("  Success, Gradient shape: ", size(g))
catch e
    println("  Failed: ", e)
end
