# Worker for the cross-version force benchmark. Run inside an environment that
# has a specific ACEpotentials version installed. Prints lines:  "<natoms> <ms>".
#
# Uses ONLY the stable public API (ace1_model -> ACEPotential calculator,
# AtomsCalculators.forces) and single-element Si so it works unchanged across
# the 0.9.x and 0.10.x series.
using ACEpotentials, AtomsBuilder, AtomsCalculators, Unitful, BenchmarkTools, Printf

label = isempty(ARGS) ? "?" : ARGS[1]

# ace1_model returns a ready-to-use ACEPotential calculator (random params).
calc = ACEpotentials.ace1_model(elements = [:Si], order = 3,
                                totaldegree = 10, rcut = 5.5)

println("# version=", pkgversion(ACEpotentials), " label=", label)
for n in (2, 3, 4)          # 16, 54, 128 atoms (cubic Si has 8/cell)
   sys = AtomsBuilder.bulk(:Si, cubic = true) * n
   rattle!(sys, 0.1u"Å")
   AtomsCalculators.forces(sys, calc)        # warmup / compile
   t = minimum(@benchmark AtomsCalculators.forces($sys, $calc) samples = 5 evals = 3).time
   @printf("%d %.6f\n", length(sys), t / 1e6)   # ms
end
