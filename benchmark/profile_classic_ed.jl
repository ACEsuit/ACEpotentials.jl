# Profile the classic analytic evaluate_ed to locate the force-path hotspot.
# Times each sub-step on a representative single-site environment.
include("common.jl")
using Printf
const P4ML = ACEpotentials.Models.P4ML
import EquivariantTensors

const s = build_model_and_calcs()
const model = s.model
const ps = s.ps
const st = s.st

# representative atomic environment (~ number of neighbours within rcut)
Nat = 40
Rs, Zs, z0 = M.rand_atenv(model, Nat)

bench(f) = (f(); minimum(@benchmark $f() samples=10 evals=5).time / 1e3) # µs

i_z0 = M._z2i(model.rbasis, z0)
rs = norm.(Rs)

# component timings
t_full_e  = bench(() -> M.evaluate(model, Rs, Zs, z0, ps, st))
t_full_ed = bench(() -> M.evaluate_ed(model, Rs, Zs, z0, ps, st))

Rnl, dRnl = M.evaluate_ed_batched(model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
Ylm, dYlm = P4ML.evaluate_ed(model.ybasis, Rs)
TA = promote_type(eltype(Rnl), eltype(Ylm))
A = zeros(TA, length(model.tensor.abasis))
EquivariantTensors.evaluate!(A, model.tensor.abasis, (Rnl, Ylm))
∂B = [ps.WB[:, i_z0]]

t_rnl_ed  = bench(() -> M.evaluate_ed_batched(model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis))
t_ylm_ed  = bench(() -> P4ML.evaluate_ed(model.ybasis, Rs))
t_tensor_eval = bench(() -> EquivariantTensors.evaluate(model.tensor, Rnl, Ylm, NamedTuple(), NamedTuple()))
t_tensor_pb   = bench(() -> EquivariantTensors.pullback(∂B, model.tensor, Rnl, Ylm, A))
t_pair_ed = model.pairbasis === nothing ? 0.0 :
            bench(() -> M.evaluate_ed_batched(model.pairbasis, rs, z0, Zs, ps.pairbasis, st.pairbasis))
alloc_ed  = (M.evaluate_ed(model, Rs, Zs, z0, ps, st); @allocated M.evaluate_ed(model, Rs, Zs, z0, ps, st))

@printf("\n=== classic evaluate_ed component profile (Nat=%d neighbours) ===\n", Nat)
@printf("evaluate (energy only)      : %8.2f µs\n", t_full_e)
@printf("evaluate_ed (energy+forces) : %8.2f µs   (%.1fx energy)\n", t_full_ed, t_full_ed/t_full_e)
@printf("  -- components --\n")
@printf("  radial evaluate_ed_batched: %8.2f µs\n", t_rnl_ed)
@printf("  Ylm   evaluate_ed         : %8.2f µs\n", t_ylm_ed)
@printf("  tensor evaluate (fwd)     : %8.2f µs\n", t_tensor_eval)
@printf("  tensor pullback (bwd)     : %8.2f µs   <-- ET migration code\n", t_tensor_pb)
@printf("  pair basis evaluate_ed    : %8.2f µs   <-- pair path\n", t_pair_ed)
@printf("  (sum of components)       : %8.2f µs\n",
        t_rnl_ed + t_ylm_ed + t_tensor_eval + t_tensor_pb + t_pair_ed)
@printf("evaluate_ed allocations     : %8d bytes\n", alloc_ed)
