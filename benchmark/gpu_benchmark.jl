# GPU Benchmark for ETACE Models
# Run with: julia --project=test benchmark/gpu_benchmark.jl
#
# Tests: ETOneBody, ETPairModel, ETACE (many-body), and combined full model

using CUDA
using LuxCUDA

using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET
using Lux, LuxCore, Random
using AtomsBase, AtomsBuilder, Unitful
using Printf

println("CUDA available: ", CUDA.functional())

# Build model
elements = (:Si, :O)
level = M.TotalDegree()
max_level = 8
order = 2
maxl = 4

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

E0s = Dict(:Si => -158.54496821, :O => -2042.0330099956639)

model = M.ace_model(; elements = elements, order = order,
            Ytype = :solid, level = level, max_level = max_level,
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal, init_Wpair = :glorot_normal,
            pair_learnable = true,
            E0s = E0s)

rng = Random.MersenneTwister(1234)
ps, st = Lux.setup(rng, model)

rcut = 5.5
NZ = 2

# ============================================================================
# 1. ETACE (many-body only)
# ============================================================================
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(rng, et_model)

# Copy parameters
for i in 1:NZ, j in 1:NZ
    idx = (i-1)*NZ + j
    et_ps.rembed.post.W[:, :, idx] .= ps.rbasis.Wnlq[:, :, i, j]
end
for iz in 1:NZ
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

# ============================================================================
# 2. ETPairModel
# ============================================================================
et_pair = ETM.convertpair(model)
pair_ps, pair_st = LuxCore.setup(rng, et_pair)

# Copy pair parameters
for i in 1:NZ, j in 1:NZ
    idx = (i-1)*NZ + j
    pair_ps.rembed.rbasis.post.W[:, :, idx] .= ps.pairbasis.Wnlq[:, :, i, j]
end
for iz in 1:NZ
    pair_ps.readout.W[1, :, iz] .= ps.Wpair[:, iz]
end

# ============================================================================
# 3. ETOneBody
# ============================================================================
zlist = ChemicalSpecies.((:Si, :O))
E0_dict = Dict(z => E0s[Symbol(z)] for z in zlist)
et_onebody = ETM.one_body(E0_dict, x -> x.z)
onebody_ps, onebody_st = LuxCore.setup(rng, et_onebody)

# GPU device
gdev = Lux.gpu_device()
println("GPU device: ", gdev)

# Benchmark configurations
configs = [
    (2, 2, 2),   # 64 atoms
    (4, 4, 4),   # 512 atoms
    (5, 5, 4),   # 800 atoms
]

println()
println("="^80)
println("GPU BENCHMARK: ETACE Models (with P4ML v0.5.8)")
println("="^80)

# ============================================================================
# SECTION 1: Many-Body Only (ETACE)
# ============================================================================
println()
println("### MANY-BODY ONLY (ETACE) - ENERGY ###")
println("| Atoms | Edges   | CPU (ms) | GPU (ms) | GPU Speedup |")
println("|-------|---------|----------|----------|-------------|")

for cfg in configs
    sys = AtomsBuilder.bulk(:Si, cubic=true) * cfg
    rattle!(sys, 0.1u"Å")
    AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
    natoms = length(sys)

    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    nedges = length(G.edge_data)

    # Warmup CPU
    _ = et_model(G, et_ps, et_st)

    # CPU benchmark
    t_cpu = @elapsed for _ in 1:10
        et_model(G, et_ps, et_st)
    end
    t_cpu_ms = (t_cpu / 10) * 1000

    # GPU setup
    G_gpu = gdev(G)
    et_ps_gpu = gdev(et_ps)
    et_st_gpu = gdev(et_st)

    # Warmup GPU
    CUDA.@sync et_model(G_gpu, et_ps_gpu, et_st_gpu)

    # GPU benchmark
    t_gpu = CUDA.@elapsed for _ in 1:10
        CUDA.@sync et_model(G_gpu, et_ps_gpu, et_st_gpu)
    end
    t_gpu_ms = (t_gpu / 10) * 1000

    speedup = t_cpu_ms / t_gpu_ms

    @printf("| %5d | %7d | %8.2f | %8.2f | %10.1fx |\n",
            natoms, nedges, t_cpu_ms, t_gpu_ms, speedup)
end

println()
println("### MANY-BODY ONLY (ETACE) - FORCES ###")
println("| Atoms | Edges   | CPU (ms) | GPU (ms) | GPU Speedup |")
println("|-------|---------|----------|----------|-------------|")

for cfg in configs
    sys = AtomsBuilder.bulk(:Si, cubic=true) * cfg
    rattle!(sys, 0.1u"Å")
    AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
    natoms = length(sys)

    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    nedges = length(G.edge_data)

    # Warmup CPU
    _ = ETM.site_grads(et_model, G, et_ps, et_st)

    # CPU benchmark
    t_cpu = @elapsed for _ in 1:5
        ETM.site_grads(et_model, G, et_ps, et_st)
    end
    t_cpu_ms = (t_cpu / 5) * 1000

    # GPU setup
    G_gpu = gdev(G)
    et_ps_gpu = gdev(et_ps)
    et_st_gpu = gdev(et_st)

    # Warmup GPU
    CUDA.@sync ETM.site_grads(et_model, G_gpu, et_ps_gpu, et_st_gpu)

    # GPU benchmark
    t_gpu = CUDA.@elapsed for _ in 1:5
        CUDA.@sync ETM.site_grads(et_model, G_gpu, et_ps_gpu, et_st_gpu)
    end
    t_gpu_ms = (t_gpu / 5) * 1000

    speedup = t_cpu_ms / t_gpu_ms

    @printf("| %5d | %7d | %8.2f | %8.2f | %10.1fx |\n",
            natoms, nedges, t_cpu_ms, t_gpu_ms, speedup)
end

# ============================================================================
# SECTION 2: Full Model (E0 + Pair + Many-Body)
# ============================================================================
println()
println("="^80)
println("### FULL MODEL (E0 + Pair + Many-Body) - ENERGY ###")
println("| Atoms | Edges   | CPU (ms) | GPU (ms) | GPU Speedup |")
println("|-------|---------|----------|----------|-------------|")

for cfg in configs
    sys = AtomsBuilder.bulk(:Si, cubic=true) * cfg
    rattle!(sys, 0.1u"Å")
    AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
    natoms = length(sys)

    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    nedges = length(G.edge_data)

    # CPU: evaluate all three models
    function full_energy_cpu(G)
        E_onebody, _ = et_onebody(G, onebody_ps, onebody_st)
        E_pair, _ = et_pair(G, pair_ps, pair_st)
        E_mb, _ = et_model(G, et_ps, et_st)
        return sum(E_onebody) + sum(E_pair) + sum(E_mb)
    end

    # Warmup CPU
    _ = full_energy_cpu(G)

    # CPU benchmark
    t_cpu = @elapsed for _ in 1:10
        full_energy_cpu(G)
    end
    t_cpu_ms = (t_cpu / 10) * 1000

    # GPU setup - all models
    G_gpu = gdev(G)
    onebody_ps_gpu = gdev(onebody_ps)
    onebody_st_gpu = gdev(onebody_st)
    pair_ps_gpu = gdev(pair_ps)
    pair_st_gpu = gdev(pair_st)
    et_ps_gpu = gdev(et_ps)
    et_st_gpu = gdev(et_st)

    function full_energy_gpu(G_gpu)
        E_onebody, _ = et_onebody(G_gpu, onebody_ps_gpu, onebody_st_gpu)
        E_pair, _ = et_pair(G_gpu, pair_ps_gpu, pair_st_gpu)
        E_mb, _ = et_model(G_gpu, et_ps_gpu, et_st_gpu)
        return sum(E_onebody) + sum(E_pair) + sum(E_mb)
    end

    # Warmup GPU
    CUDA.@sync full_energy_gpu(G_gpu)

    # GPU benchmark
    t_gpu = CUDA.@elapsed for _ in 1:10
        CUDA.@sync full_energy_gpu(G_gpu)
    end
    t_gpu_ms = (t_gpu / 10) * 1000

    speedup = t_cpu_ms / t_gpu_ms

    @printf("| %5d | %7d | %8.2f | %8.2f | %10.1fx |\n",
            natoms, nedges, t_cpu_ms, t_gpu_ms, speedup)
end

println()
println("### FULL MODEL (E0 + Pair + Many-Body) - FORCES ###")
println("| Atoms | Edges   | CPU (ms) | GPU (ms) | GPU Speedup |")
println("|-------|---------|----------|----------|-------------|")

for cfg in configs
    sys = AtomsBuilder.bulk(:Si, cubic=true) * cfg
    rattle!(sys, 0.1u"Å")
    AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
    natoms = length(sys)

    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    nedges = length(G.edge_data)

    # CPU: evaluate all three gradients
    function full_grads_cpu(G)
        ∂onebody = ETM.site_grads(et_onebody, G, onebody_ps, onebody_st)
        ∂pair = ETM.site_grads(et_pair, G, pair_ps, pair_st)
        ∂mb = ETM.site_grads(et_model, G, et_ps, et_st)
        return (∂onebody, ∂pair, ∂mb)
    end

    # Warmup CPU
    _ = full_grads_cpu(G)

    # CPU benchmark
    t_cpu = @elapsed for _ in 1:5
        full_grads_cpu(G)
    end
    t_cpu_ms = (t_cpu / 5) * 1000

    # GPU setup - all models
    G_gpu = gdev(G)
    onebody_ps_gpu = gdev(onebody_ps)
    onebody_st_gpu = gdev(onebody_st)
    pair_ps_gpu = gdev(pair_ps)
    pair_st_gpu = gdev(pair_st)
    et_ps_gpu = gdev(et_ps)
    et_st_gpu = gdev(et_st)

    function full_grads_gpu(G_gpu)
        ∂onebody = ETM.site_grads(et_onebody, G_gpu, onebody_ps_gpu, onebody_st_gpu)
        ∂pair = ETM.site_grads(et_pair, G_gpu, pair_ps_gpu, pair_st_gpu)
        ∂mb = ETM.site_grads(et_model, G_gpu, et_ps_gpu, et_st_gpu)
        return (∂onebody, ∂pair, ∂mb)
    end

    # Warmup GPU
    CUDA.@sync full_grads_gpu(G_gpu)

    # GPU benchmark
    t_gpu = CUDA.@elapsed for _ in 1:5
        CUDA.@sync full_grads_gpu(G_gpu)
    end
    t_gpu_ms = (t_gpu / 5) * 1000

    speedup = t_cpu_ms / t_gpu_ms

    @printf("| %5d | %7d | %8.2f | %8.2f | %10.1fx |\n",
            natoms, nedges, t_cpu_ms, t_gpu_ms, speedup)
end

println()
