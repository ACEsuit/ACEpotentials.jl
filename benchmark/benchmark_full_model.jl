# Benchmark: Full model (1+2+many body) with StackedCalculator
# Compares ACE CPU vs ETACE CPU vs ETACE GPU for energy and forces

using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET
import AtomsCalculators
using StaticArrays, Lux, Random, LuxCore, LinearAlgebra
using AtomsBase, AtomsBuilder, Unitful
using BenchmarkTools
using Printf

# GPU detection
dev = identity
has_cuda = false

try
   using CUDA
   if CUDA.functional()
      @info "Using CUDA"
      CUDA.versioninfo()
      global has_cuda = true
      global dev = cu
   else
      @info "CUDA is not functional"
   end
catch e
   @info "Couldn't load CUDA: $e"
end

if !has_cuda
   @info "No GPU available. Using CPU only."
end

rng = Random.MersenneTwister(1234)

# Build models with E0s and pair potential enabled
elements = (:Si, :O)
level = M.TotalDegree()
max_level = 8
order = 2
maxl = 4

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

# E0s for one-body
E0s = Dict(:Si => -158.54496821, :O => -2042.0330099956639)

model = M.ace_model(; elements = elements, order = order,
            Ytype = :solid, level = level, max_level = max_level,
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal, init_Wpair = :glorot_normal,
            pair_learnable = true,  # Keep learnable for ET conversion
            E0s = E0s)

ps, st = Lux.setup(rng, model)

# Create old ACE calculator (full model with E0s and pair)
ace_calc = M.ACEPotential(model, ps, st)

# Convert to full ETACE with StackedCalculator
et_calc = ETM.convert2et_full(model, ps, st)

rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

# Function to create system of given size
function make_system(n_repeat)
    sys = AtomsBuilder.bulk(:Si, cubic=true) * n_repeat
    rattle!(sys, 0.1u"Å")
    AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
    return sys
end

# Benchmark configurations
configs = [
    (2, 2, 2),   # 64 atoms
    (3, 3, 2),   # 144 atoms
    (4, 4, 2),   # 256 atoms
    (4, 4, 4),   # 512 atoms
    (5, 5, 4),   # 800 atoms
]

println()
println("=" ^ 90)
println("BENCHMARK: Full Model (1+2+many body) - ACE vs ETACE StackedCalculator")
println("=" ^ 90)
println()

# --- ENERGY BENCHMARK ---
println("### ENERGY ###")
println()

if has_cuda
    println("| Atoms | Edges   | ACE CPU (ms) | ETACE CPU (ms) | ETACE GPU (ms) | CPU Speedup | GPU Speedup |")
    println("|-------|---------|--------------|----------------|----------------|-------------|-------------|")
else
    println("| Atoms | Edges   | ACE CPU (ms) | ETACE CPU (ms) | CPU Speedup |")
    println("|-------|---------|--------------|----------------|-------------|")
end

for cfg in configs
    sys = make_system(cfg)
    natoms = length(sys)

    # Count edges
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    nedges = length(G.edge_data)

    # Warmup ACE
    _ = AtomsCalculators.potential_energy(sys, ace_calc)

    # Warmup ETACE CPU
    _ = AtomsCalculators.potential_energy(sys, et_calc)

    # Benchmark ACE CPU
    t_ace = @belapsed AtomsCalculators.potential_energy($sys, $ace_calc) samples=5 evals=3
    t_ace_ms = t_ace * 1000

    # Benchmark ETACE CPU
    t_etace_cpu = @belapsed AtomsCalculators.potential_energy($sys, $et_calc) samples=5 evals=3
    t_etace_cpu_ms = t_etace_cpu * 1000

    cpu_speedup = t_ace_ms / t_etace_cpu_ms

    if has_cuda
        # For GPU we need to handle the StackedCalculator with GPU-capable models
        # TODO: GPU version of StackedCalculator
        t_etace_gpu_ms = NaN
        gpu_speedup = NaN

        @printf("| %5d | %7d | %12.2f | %14.2f | %14s | %10.1fx | %10s |\n",
                natoms, nedges, t_ace_ms, t_etace_cpu_ms, "N/A", cpu_speedup, "N/A")
    else
        @printf("| %5d | %7d | %12.2f | %14.2f | %10.1fx |\n",
                natoms, nedges, t_ace_ms, t_etace_cpu_ms, cpu_speedup)
    end
end

println()

# --- FORCES BENCHMARK ---
println("### FORCES ###")
println()

if has_cuda
    println("| Atoms | Edges   | ACE CPU (ms) | ETACE CPU (ms) | ETACE GPU (ms) | CPU Speedup | GPU Speedup |")
    println("|-------|---------|--------------|----------------|----------------|-------------|-------------|")
else
    println("| Atoms | Edges   | ACE CPU (ms) | ETACE CPU (ms) | CPU Speedup |")
    println("|-------|---------|--------------|----------------|-------------|")
end

for cfg in configs
    sys = make_system(cfg)
    natoms = length(sys)

    # Count edges
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    nedges = length(G.edge_data)

    # Warmup ACE
    _ = AtomsCalculators.forces(sys, ace_calc)

    # Warmup ETACE CPU
    _ = AtomsCalculators.forces(sys, et_calc)

    # Benchmark ACE CPU
    t_ace = @belapsed AtomsCalculators.forces($sys, $ace_calc) samples=5 evals=3
    t_ace_ms = t_ace * 1000

    # Benchmark ETACE CPU
    t_etace_cpu = @belapsed AtomsCalculators.forces($sys, $et_calc) samples=5 evals=3
    t_etace_cpu_ms = t_etace_cpu * 1000

    cpu_speedup = t_ace_ms / t_etace_cpu_ms

    if has_cuda
        t_etace_gpu_ms = NaN
        gpu_speedup = NaN

        @printf("| %5d | %7d | %12.2f | %14.2f | %14s | %10.1fx | %10s |\n",
                natoms, nedges, t_ace_ms, t_etace_cpu_ms, "N/A", cpu_speedup, "N/A")
    else
        @printf("| %5d | %7d | %12.2f | %14.2f | %10.1fx |\n",
                natoms, nedges, t_ace_ms, t_etace_cpu_ms, cpu_speedup)
    end
end

println()
println("Notes:")
println("- ACE CPU: Original ACEpotentials model (full: E0 + pair + many-body)")
println("- ETACE CPU: StackedCalculator with ETOneBody + ETPairModel + ETACE")
println("- CPU Speedup = ACE CPU / ETACE CPU")
println("- Graph construction time included in ETACE timings")
