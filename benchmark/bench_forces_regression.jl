# Force-evaluation performance benchmark: classic (analytic) vs ET backend.
#
# Background: a ~13x force-evaluation regression was reported in the v0.10 (ET)
# series vs the previous release. The original hypothesis was that the ET
# backend's Zygote autograd forces (`site_grads`) were the cause. This benchmark
# (together with benchmark/bench_crossversion.jl) REFUTED that: the ET autograd
# path is actually faster than the classic path was. The real cause was a TYPE
# INSTABILITY in the classic analytic `evaluate_ed` (src/models/ace.jl):
# `EquivariantTensors.pullback` returns `Tuple{Any,Any}`, so the inline gradient
# assembly ran with per-element dynamic dispatch (~10x slower). Fixed with a
# function barrier (`_assemble_grad_ed!`).
#
# This script measures, for parameter-matched classic vs ET calculators on the
# SAME systems: potential_energy and forces (time + allocations), plus an
# isolation timing of the gradient step. It is kept as a backend comparison and
# a manual regression check (the automated guard is benchmark/benchmarks.jl).
#
# Run:  julia --project=benchmark benchmark/bench_forces_regression.jl

include("common.jl")
using Printf

const setup = build_model_and_calcs()
const ace_calc = setup.ace_calc
const et_calc  = setup.et_calc
const rcut     = setup.rcut

# ---------------------------------------------------------------------------
# 1c. Correctness gate: ET forces must match classic forces before timing.
# ---------------------------------------------------------------------------
function check_consistency()
   sys = make_system(SIZE_CONFIGS[1])
   Ec = AtomsCalculators.potential_energy(sys, ace_calc)
   Ee = AtomsCalculators.potential_energy(sys, et_calc)
   Fc = AtomsCalculators.forces(sys, ace_calc)
   Fe = AtomsCalculators.forces(sys, et_calc)
   eE = abs(ustrip(Ee - Ec)) / max(abs(ustrip(Ec)), 1)
   maxF = maximum(norm.(ustrip.(Fc)))
   eF = maximum(norm.(ustrip.(Fe .- Fc))) / max(maxF, 1)
   @printf("Consistency check (%d atoms):  ΔE/|E| = %.2e   max|ΔF|/max|F| = %.2e\n",
           length(sys), eE, eF)
   if eE > 1e-6 || eF > 1e-6
      @warn "Classic and ET results disagree beyond 1e-6 — timings of an " *
            "incorrect path are meaningless." eE eF
   end
   return eE, eF
end

# ---------------------------------------------------------------------------
# 1a. Head-to-head energy & forces benchmark.
# ---------------------------------------------------------------------------
function run_table()
   println("\n", "="^104)
   println("FORCE REGRESSION BENCHMARK: classic (analytic) vs ET (Zygote autograd)")
   println("="^104, "\n")
   println("| Atoms | Edges | E classic (ms) | E ET (ms) | E ratio | F classic (ms) | F ET (ms) | F ratio | F ET allocs |")
   println("|-------|-------|----------------|-----------|---------|----------------|-----------|---------|-------------|")

   results = Vector{Any}()
   for cfg in SIZE_CONFIGS
      sys = make_system(cfg)
      nat = length(sys)
      ne  = nedges(sys, rcut)

      eC = measure(() -> AtomsCalculators.potential_energy(sys, ace_calc))
      eE = measure(() -> AtomsCalculators.potential_energy(sys, et_calc))
      fC = measure(() -> AtomsCalculators.forces(sys, ace_calc))
      fE = measure(() -> AtomsCalculators.forces(sys, et_calc))

      e_ratio = eE.time_ns / eC.time_ns
      f_ratio = fE.time_ns / fC.time_ns

      @printf("| %5d | %5d | %14.3f | %9.3f | %7.2f | %14.3f | %9.3f | %7.2f | %11d |\n",
              nat, ne, eC.time_ns/1e6, eE.time_ns/1e6, e_ratio,
              fC.time_ns/1e6, fE.time_ns/1e6, f_ratio, fE.allocs)

      push!(results, (; natoms = nat, nedges = ne,
                      e_classic_ns = eC.time_ns, e_et_ns = eE.time_ns, e_ratio,
                      f_classic_ns = fC.time_ns, f_et_ns = fE.time_ns, f_ratio,
                      f_et_allocs = fE.allocs, f_et_mem = fE.memory,
                      e_et_allocs = eE.allocs, e_et_mem = eE.memory))
   end
   return results
end

# ---------------------------------------------------------------------------
# Isolation: time the gradient step itself (Zygote `site_grads`) vs the
# analytic `site_basis_jacobian`, with the graph precomputed — isolates the AD
# cost from graph construction and force assembly.
# ---------------------------------------------------------------------------
function run_isolation()
   println("\nIsolation: gradient step only (graph precomputed)")
   println("| Atoms | site_grads (Zygote, ms) | site_basis_jacobian (analytic, ms) | ratio |")
   println("|-------|-------------------------|------------------------------------|-------|")
   # Reach into the StackedCalculator to grab the ETACE many-body component.
   ace_sub = et_calc.calcs[end]   # ETACE WrappedSiteCalculator (last in the stack)
   m, ps, st = ace_sub.model, ace_sub.ps, ace_sub.st
   for cfg in SIZE_CONFIGS[1:min(3, end)]
      sys = make_system(cfg)
      G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
      tg = measure(() -> ETM.site_grads(m, G, ps, st))
      tj = measure(() -> ETM.site_basis_jacobian(m, G, ps, st))
      @printf("| %5d | %23.3f | %34.3f | %5.2f |\n",
              length(sys), tg.time_ns/1e6, tj.time_ns/1e6, tg.time_ns/tj.time_ns)
   end
end

check_consistency()
results = run_table()
run_isolation()

# --- summary ---
mean_e = sum(r.e_ratio for r in results) / length(results)
mean_f = sum(r.f_ratio for r in results) / length(results)
mean_fe_classic = sum(r.f_classic_ns / r.e_classic_ns for r in results) / length(results)
println("\nMean energy ratio (ET/classic): ", round(mean_e, digits=2))
println("Mean force  ratio (ET/classic): ", round(mean_f, digits=2))
println("Mean classic force/energy ratio: ", round(mean_fe_classic, digits=2),
        "  (was ~12-13x before the evaluate_ed type-stability fix; ~1.5x after)")
