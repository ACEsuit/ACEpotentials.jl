using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET
import AtomsCalculators
using StaticArrays, Lux, Random, LuxCore, LinearAlgebra
using AtomsBase, AtomsBuilder, Unitful
using BenchmarkTools
using Printf

# GPU detection (simplified from ET test utils)
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

# Build models
elements = (:Si, :O)
level = M.TotalDegree()
max_level = 8
order = 2
maxl = 4

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

model = M.ace_model(; elements = elements, order = order,
            Ytype = :solid, level = level, max_level = max_level,
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, model)

# Kill the pair basis for fair comparison
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0
end

# Create old ACE calculator
ace_calc = M.ACEPotential(model, ps, st)

# Convert to ETACE
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(rng, et_model)

# Copy parameters
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.rembed.post.W[:, :, 2] = ps.rbasis.Wnlq[:, :, 1, 2]
et_ps.rembed.post.W[:, :, 3] = ps.rbasis.Wnlq[:, :, 2, 1]
et_ps.rembed.post.W[:, :, 4] = ps.rbasis.Wnlq[:, :, 2, 2]
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]
et_ps.readout.W[1, :, 2] .= ps.WB[:, 2]

rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

# GPU setup
has_gpu = has_cuda
if has_gpu
    et_ps_32 = ET.float32(et_ps)
    et_st_32 = ET.float32(et_st)
    et_ps_gpu = dev(et_ps_32)
    et_st_gpu = dev(et_st_32)
end

# Function to create system of given size
function make_system(n_repeat)
    sys = AtomsBuilder.bulk(:Si, cubic=true) * n_repeat
    rattle!(sys, 0.1u"Å")
    AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
    return sys
end

# Benchmark configurations (tuple for bulk multiplication)
configs = [
    (1, 1, 1),   # 8 atoms
    (2, 1, 1),   # 16 atoms
    (2, 2, 2),   # 64 atoms
    (3, 3, 2),   # 144 atoms
    (4, 4, 2),   # 256 atoms
    (4, 4, 4),   # 512 atoms
    (5, 5, 4),   # 800 atoms
]

println()
println("=" ^ 85)
println("BENCHMARK: ACE (CPU) vs ETACE (CPU) vs ETACE (GPU)")
println("=" ^ 85)
println()

# Header
if has_gpu
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
    _ = sum(et_model(G, et_ps, et_st)[1])

    # Benchmark ACE CPU
    t_ace = @belapsed AtomsCalculators.potential_energy($sys, $ace_calc) samples=5 evals=3
    t_ace_ms = t_ace * 1000

    # Benchmark ETACE CPU (graph construction NOT included for fair comparison with GPU)
    t_etace_cpu = @belapsed sum($et_model($G, $et_ps, $et_st)[1]) samples=5 evals=3
    t_etace_cpu_ms = t_etace_cpu * 1000

    cpu_speedup = t_ace_ms / t_etace_cpu_ms

    if has_gpu
        # Convert graph to GPU (Float32)
        G_32 = ET.float32(G)
        G_gpu = dev(G_32)

        # Warmup GPU
        _ = sum(et_model(G_gpu, et_ps_gpu, et_st_gpu)[1])

        # Benchmark ETACE GPU (graph already on GPU)
        t_etace_gpu = @belapsed sum($et_model($G_gpu, $et_ps_gpu, $et_st_gpu)[1]) samples=5 evals=3
        t_etace_gpu_ms = t_etace_gpu * 1000

        gpu_speedup = t_ace_ms / t_etace_gpu_ms

        @printf("| %5d | %7d | %12.2f | %14.2f | %14.2f | %10.1fx | %10.1fx |\n",
                natoms, nedges, t_ace_ms, t_etace_cpu_ms, t_etace_gpu_ms, cpu_speedup, gpu_speedup)
    else
        @printf("| %5d | %7d | %12.2f | %14.2f | %10.1fx |\n",
                natoms, nedges, t_ace_ms, t_etace_cpu_ms, cpu_speedup)
    end
end

println()
println("Notes:")
println("- ACE CPU: Original ACEpotentials model (pair basis zeroed for fair comparison)")
println("- ETACE CPU: EquivariantTensors backend on CPU (Float64)")
println("- ETACE GPU: EquivariantTensors backend on GPU (Float32)")
println("- CPU Speedup = ACE CPU / ETACE CPU")
println("- GPU Speedup = ACE CPU / ETACE GPU")
println("- Graph construction time NOT included (currently CPU-only)")
