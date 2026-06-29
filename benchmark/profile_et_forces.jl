# Profile the ET ACE force path to find optimization scope.
# Breaks the stacked-calculator forces into components, and the ETACE component
# into graph-build / site_grads(Zygote) / forces_from_edge_grads, with allocations.
include("common.jl")
using Printf
import EquivariantTensors as ET

s = build_model_and_calcs()
et_calc = s.et_calc          # StackedCalculator(onebody, pair, ace)
rcut    = s.rcut
onebody, pair, ace = et_calc.calcs

bench(f) = (f(); minimum(@benchmark $f() samples=10 evals=3).time/1e6)   # ms
allocOf(f) = (f(); @allocated f())

for cfg in SIZE_CONFIGS[[1, 3, 5]]
   sys = make_system(cfg)
   nat = length(sys)
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   ne = length(G.edge_data)

   t_full = bench(() -> AtomsCalculators.forces(sys, et_calc))
   t_1b   = bench(() -> AtomsCalculators.forces(sys, onebody))
   t_pair = bench(() -> AtomsCalculators.forces(sys, pair))
   t_ace  = bench(() -> AtomsCalculators.forces(sys, ace))

   m, ps, st = ace.model, ace.ps, ace.st
   t_graph = bench(() -> ET.Atoms.interaction_graph(sys, rcut * u"Å"))
   t_sg    = bench(() -> ETM.site_grads(m, G, ps, st))
   ∂G = ETM.site_grads(m, G, ps, st)
   t_ffe   = bench(() -> ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data))
   a_sg    = allocOf(() -> ETM.site_grads(m, G, ps, st))

   @printf("\n=== %d atoms, %d edges ===\n", nat, ne)
   @printf("full stacked forces      : %8.3f ms\n", t_full)
   @printf("  onebody forces         : %8.3f ms\n", t_1b)
   @printf("  pair forces            : %8.3f ms\n", t_pair)
   @printf("  ETACE forces           : %8.3f ms\n", t_ace)
   @printf("    -- ETACE breakdown --\n")
   @printf("    graph build          : %8.3f ms\n", t_graph)
   @printf("    site_grads           : %8.3f ms   alloc=%d B\n", t_sg, a_sg)
   @printf("    forces_from_edge_grads: %7.3f ms\n", t_ffe)
end
