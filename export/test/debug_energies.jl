using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACEPotential
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
import AtomsBase
using Unitful
using AtomsCalculators
import EquivariantTensors as ET

const M = ACEpotentials.Models

# Create splinified model
elements = (:Si,)
rcut = 5.5

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = 2,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = 8,
    maxl = 2,
    pair_maxn = 8,
    rin0cuts = rin0cuts,
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

et_model = ACEpotentials.ETModels.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

n_species = length(elements)
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
end
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

# Splinify
et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

for iz in 1:n_species
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

calc_splined = ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

# Create test system
a0 = 5.43
positions = [
    SVector(0.0, 0.0, 0.0),
    SVector(a0/4, a0/4, a0/4),
    SVector(a0/2, a0/2, 0.0),
    SVector(a0/2, 0.0, a0/2),
]
box = [SVector(a0, 0.0, 0.0), SVector(0.0, a0, 0.0), SVector(0.0, 0.0, a0)]

sys = AtomsBase.periodic_system(
    [:Si => pos * u"Å" for pos in positions],
    [b * u"Å" for b in box]
)

# Get total energy from reference
E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_splined))
println("Reference total energy: $E_ref eV")

# Build neighbor lists
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

neighbor_data = Vector{Tuple{Vector{SVector{3,Float64}}, Vector{Int}}}()
for i in 1:4
    Rs_i = SVector{3, Float64}[]
    Zs_i = Int[]
    for (edge_idx, edge) in enumerate(G.edge_data)
        if G.ii[edge_idx] == i
            push!(Rs_i, SVector{3, Float64}(edge.𝐫))
            push!(Zs_i, Int(edge.z1.atomic_number))
        end
    end
    push!(neighbor_data, (Rs_i, Zs_i))
end

println("\n=== Neighbors per atom ===")
for (i, (Rs, Zs)) in enumerate(neighbor_data)
    println("Atom $i: $(length(Rs)) neighbors")
    for (j, R) in enumerate(Rs)
        println("  R[$j] = $R, |R| = $(norm(R)), Z = $(Zs[j])")
    end
end

# Include exported model
println("\n=== Including exported model ===")
exp_mod = Module(:ExportedModel)
Base.include(exp_mod, joinpath(@__DIR__, "build", "hermite_accuracy_test.jl"))

# Evaluate site energies
println("\n=== Site energies ===")
site_energies = Float64[]
for (i, (Rs, Zs)) in enumerate(neighbor_data)
    if length(Rs) > 0
        E_i = exp_mod.site_energy(Rs, Zs, 14)
        println("Atom $i: E = $E_i eV")
        push!(site_energies, E_i)
    else
        println("Atom $i: no neighbors")
        push!(site_energies, 0.0)
    end
end

E_total_exp = sum(site_energies)
println("\nExported total energy: $E_total_exp eV")
println("Reference total energy: $E_ref eV")
println("Ratio ref/exp: ", E_ref / E_total_exp)

# Compare Rnl directly at r=2.35
println("\n=== Comparing Rnl at r=2.35 ===")

r_test = 2.35

# Manually evaluate spline from reference model data
rembed_layer = et_model_splined.rembed.layer
rembed_st = et_st_splined.rembed.params

# Get transform parameters from reference
trans = rembed_layer.trans
trans_st = trans.refstate
agnesi_p = trans_st.params[1]

# Agnesi transform
a = agnesi_p.a
b0 = agnesi_p.b0
b1 = agnesi_p.b1
req = agnesi_p.req

x = (r_test - req) / a
u = b0 + b1 * x
σ = 1.0 / (1.0 + exp(u))
y_ref = 2σ - 1.0
println("Reference y = ", y_ref)

# Get spline data
refstate = rembed_st
F_ref = refstate.F
G_ref = refstate.G
y_min = refstate.x0[1]
y_max = refstate.x1[1]
n_knots = size(F_ref, 1)
h = (y_max - y_min) / (n_knots - 1)

println("Ref spline grid: y_min=$y_min, y_max=$y_max, n_knots=$n_knots, h=$h")

# Evaluate spline
t_raw = (y_ref - y_min) / h
t_frac, t_floor = modf(t_raw)
il = trunc(Int, t_floor) + 1
il = clamp(il, 1, n_knots - 1)

fl_ref = F_ref[il, 1]  # SVector{19, Float64}
fr_ref = F_ref[il+1, 1]
gl_ref = h .* G_ref[il, 1]
gr_ref = h .* G_ref[il+1, 1]

a0 = fl_ref
a1 = gl_ref
a2 = @. -3fl_ref + 3fr_ref - 2gl_ref - gr_ref
a3 = @. 2fl_ref - 2fr_ref + gl_ref + gr_ref
s_ref = @. ((a3 * t_frac + a2) * t_frac + a1) * t_frac + a0

# Y-space envelope
env_y = (1.0 - y_ref^2)^2
Rnl_ref_manual = env_y .* s_ref

println("\nReference (manual calculation):")
println("  s_ref[1:5] = ", s_ref[1:5])
println("  env_y = ", env_y)
println("  Rnl_ref[1:5] = ", Rnl_ref_manual[1:5])

# Get Rnl from exported model
Rnl_exp = exp_mod.evaluate_Rnl(r_test, 1, 1)
println("\nExported:")
println("  Rnl_exp[1:5] = ", Rnl_exp[1:5])

println("\nRatio ref/exp for Rnl = ", Rnl_ref_manual[1:5] ./ Rnl_exp[1:5])
println("Are they equal? ", Rnl_ref_manual ≈ Rnl_exp)

# Check model structure
println("\n=== Model structure ===")
println("et_model_splined type: ", typeof(et_model_splined))
println("et_model_splined fields: ", fieldnames(typeof(et_model_splined)))

println("\nrembed layer type: ", typeof(et_model_splined.rembed))
if hasproperty(et_model_splined.rembed, :layer)
    println("  rembed.layer type: ", typeof(et_model_splined.rembed.layer))
end
if hasproperty(et_model_splined.rembed, :post)
    println("  rembed.post type: ", typeof(et_model_splined.rembed.post))
end

# Check if there's a post-processing after splines
println("\net_ps_splined structure:")
for (k, v) in pairs(et_ps_splined)
    println("  $k: ", typeof(v))
    if isa(v, NamedTuple)
        for (k2, v2) in pairs(v)
            if isa(v2, AbstractArray)
                println("    $k2: ", typeof(v2), " size=", size(v2))
            else
                println("    $k2: ", typeof(v2))
            end
        end
    end
end

# Check rembed post weights
if haskey(et_ps_splined, :rembed) && hasproperty(et_ps_splined.rembed, :post)
    W_post = et_ps_splined.rembed.post.W
    println("\nrembed.post.W: ", typeof(W_post), " size=", size(W_post))
    println("W_post[1:3, 1:3, 1] = ")
    println(W_post[1:3, 1:3, 1])
end

# Check unsplinified model parameters
println("\n=== Unsplinified model parameters ===")
for (k, v) in pairs(et_ps)
    println("  $k: ", typeof(v))
    if isa(v, NamedTuple)
        for (k2, v2) in pairs(v)
            if isa(v2, NamedTuple)
                for (k3, v3) in pairs(v2)
                    if isa(v3, AbstractArray)
                        println("    $k2.$k3: ", typeof(v3), " size=", size(v3))
                    end
                end
            elseif isa(v2, AbstractArray)
                println("    $k2: ", typeof(v2), " size=", size(v2))
            end
        end
    end
end

# Compare Ylm
println("\n=== Comparing Ylm ===")
R_dir = SVector(1.0, 0.0, 0.0)  # Unit vector along x
Ylm_exp = exp_mod.eval_ylm(R_dir)
println("Exported Ylm (R=[1,0,0]): ", Ylm_exp)

# Check ABASIS_SPEC mapping
println("\n=== ABASIS_SPEC (exported) ===")
println("First 10 entries: ", exp_mod.ABASIS_SPEC[1:10])

# Check sizes
println("\n=== Exported model dimensions ===")
println("N_RNL = ", exp_mod.N_RNL)
println("N_YLM = ", exp_mod.N_YLM)
println("N_BASIS = ", exp_mod.N_BASIS)

# Compare B-basis for atom 1
println("\n=== Comparing B-basis for atom 1 ===")
Rs1, Zs1 = neighbor_data[1]
B_exp = exp_mod.site_basis(Rs1, Zs1, 14)
println("Exported B[1:10] = ", B_exp[1:10])
println("Sum of B_exp = ", sum(B_exp))
println("max(abs(B_exp)) = ", maximum(abs.(B_exp)))

# Evaluate WB to see contribution
WB_1 = exp_mod.WB_1
println("\nExported WB[1:10] = ", WB_1[1:10])
println("dot(B, WB) = ", dot(B_exp, WB_1))

# Compare with direct site_energy result
E1_direct = exp_mod.site_energy(Rs1, Zs1, 14)
println("\nsite_energy result = ", E1_direct, " eV")

# Check A2B coupling matrix
println("\n=== A2B coupling matrix (exported) ===")
println("A2BMAP_1_I (first 20): ", exp_mod.A2BMAP_1_I[1:20])
println("A2BMAP_1_J (first 20): ", exp_mod.A2BMAP_1_J[1:20])
println("A2BMAP_1_V (first 20): ", exp_mod.A2BMAP_1_V[1:20])
println("A2BMAP_1_SIZE: ", exp_mod.A2BMAP_1_SIZE)

# Check reference A2B
basis_ref = et_model_splined.basis
println("\nReference basis type: ", typeof(basis_ref))
println("Reference basis fields: ", fieldnames(typeof(basis_ref)))

# Use correct field names
println("Reference basis.A2Bmaps: ", typeof(basis_ref.A2Bmaps))
if length(basis_ref.A2Bmaps) > 0
    a2b_ref = basis_ref.A2Bmaps[1]
    println("A2B matrix size: ", size(a2b_ref))
    println("A2B nonzeros: ", count(!iszero, a2b_ref))
end

# Get reference AA-basis spec
aa_ref = basis_ref.aabasis
println("\nReference AA basis type: ", typeof(aa_ref))
if hasproperty(aa_ref, :specs)
    specs = aa_ref.specs
    println("AA specs type: ", typeof(specs))
    if isa(specs, Tuple)
        for (i, sp) in enumerate(specs)
            println("  Order $i: $(length(sp)) terms")
            if length(sp) > 0
                println("    First 3: ", sp[1:min(3, length(sp))])
            end
        end
    end
end

# Check reference A-basis spec
a_ref = basis_ref.abasis
println("\nReference A-basis type: ", typeof(a_ref))
if hasproperty(a_ref, :spec)
    a_spec = a_ref.spec
    println("A spec length: ", length(a_spec))
    println("A spec first 10: ", a_spec[1:min(10, length(a_spec))])
end

# Compare exported vs reference A-basis spec
println("\n=== A-basis comparison ===")
println("Exported ABASIS_SPEC length: ", length(exp_mod.ABASIS_SPEC))
println("Exported: ", exp_mod.ABASIS_SPEC[1:min(10, length(exp_mod.ABASIS_SPEC))])

# Compare exported vs reference AA-basis spec
println("\n=== AA-basis comparison ===")
println("Exported AABASIS_SPECS_1: ", exp_mod.AABASIS_SPECS_1)
println("Exported AABASIS_SPECS_2 (first 5): ", exp_mod.AABASIS_SPECS_2[1:min(5, length(exp_mod.AABASIS_SPECS_2))])

# Try to directly evaluate reference model site_energy
println("\n=== Direct reference evaluation ===")

# The ETACEPotential wraps the model
# Let's look at how it evaluates site energy
println("calc_splined type: ", typeof(calc_splined))
println("calc_splined.model type: ", typeof(calc_splined.model))

# Check if there's a direct site_energy function
println("\nTrying to evaluate site energy for single site...")

# Let's check how many atoms and calculate per-atom energy
E_ref_total = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_splined))
n_atoms = length(sys)
E_ref_per_atom = E_ref_total / n_atoms
println("Reference total: $E_ref_total eV for $n_atoms atoms")
println("Reference per atom: $E_ref_per_atom eV")

# Check if there's a site_energy equivalent
# Looking at the forces - if forces are per atom, maybe energies are accumulated differently
F_ref = AtomsCalculators.forces(sys, calc_splined)
println("\nReference forces per atom:")
for i in 1:n_atoms
    println("  F[$i] = ", ustrip.(u"eV/Å", F_ref[i]))
end

# Test with a simple single-neighbor case
println("\n=== Single neighbor test ===")
R_single = [SVector(2.35, 0.0, 0.0)]  # One neighbor at 2.35 Angstrom along x
Z_single = [14]

E_single = exp_mod.site_energy(R_single, Z_single, 14)
println("Energy with single neighbor at r=2.35 along x: $E_single eV")

# Get the intermediate values
Rnl, Ylm = exp_mod.compute_embeddings(R_single, Z_single, 14)
println("Rnl[1, 1:5] = ", [Rnl[1, i] for i in 1:5])
println("Ylm[1, :] = ", [Ylm[1, i] for i in 1:exp_mod.N_YLM])

# Compute A-basis manually
A_manual = zeros(Float64, length(exp_mod.ABASIS_SPEC))
for (iA, (n, l)) in enumerate(exp_mod.ABASIS_SPEC)
    A_manual[iA] = Rnl[1, n] * Ylm[1, l]
end
println("A-basis[1:10] = ", A_manual[1:10])
println("Sum(A) = ", sum(A_manual))

# Now trace through tensor_evaluate
B_single, A_single = exp_mod.tensor_evaluate(Rnl, Ylm)
println("B-basis (from tensor_evaluate) [1:10] = ", B_single[1:10])
println("dot(B, WB) = ", dot(B_single, exp_mod.WB_1))

# Try to evaluate reference model directly for the same single neighbor case
println("\n=== Reference model evaluation for single neighbor ===")

# Create edge data for single neighbor
using DecoratedParticles: PState
const DP = DecoratedParticles

# Create a minimal system with one atom and one neighbor for reference eval
R_test = SVector(2.35, 0.0, 0.0)

# Get rembed evaluation from reference
rembed = et_model_splined.rembed
yembed = et_model_splined.yembed

# Build edge data
edge = (𝐫 = R_test, z0 = AtomsBase.ChemicalSpecies(:Si), z1 = AtomsBase.ChemicalSpecies(:Si))

# Evaluate Rnl through reference model
edges = [edge]
println("edge = ", edge)

# Try direct evaluation through the model layers
# The rembed layer is TransSelSplines
rembed_layer = et_model_splined.rembed.layer
println("\nrembed_layer type: ", typeof(rembed_layer))

# Evaluate Rnl for the edge
# TransSelSplines evaluation takes (x, ps, st) where x is the edge
# Let's try to evaluate manually

# Get reference Rnl from the model
# rembed is EdgeEmbed{TransSelSplines{...}}
# EdgeEmbed wraps the layer and evaluates it on each edge

# For reference, let me evaluate using EquivariantTensors directly
using EquivariantTensors

# Single edge - get R values
# The reference model uses edge.𝐫 for position
r_ref = norm(edge.𝐫)
rhat_ref = edge.𝐫 / r_ref
println("r_ref = $r_ref, rhat = $rhat_ref")

# Try to directly use EquivariantTensors functions
# The basis field is SparseACEbasis
basis = et_model_splined.basis
println("\nbasis type: ", typeof(basis))
println("basis.abasis type: ", typeof(basis.abasis))
println("basis.aabasis type: ", typeof(basis.aabasis))

# Compare A-basis spec
println("\n=== Reference vs Exported A-basis spec ===")
ref_aspec = et_st_splined.basis.aspec
println("Reference aspec length: ", length(ref_aspec))
println("Reference aspec: ", ref_aspec)
println("\nExported ABASIS_SPEC: ", exp_mod.ABASIS_SPEC)
println("Match? ", ref_aspec == collect(exp_mod.ABASIS_SPEC))

# Compare AA-basis spec
println("\n=== Reference vs Exported AA-basis spec ===")
ref_aaspecs = et_st_splined.basis.aaspecs
println("Reference aaspecs: ", ref_aaspecs)
println("\nExported AABASIS_SPECS_1: ", exp_mod.AABASIS_SPECS_1)
println("Exported AABASIS_SPECS_2: ", exp_mod.AABASIS_SPECS_2)
println("Order 1 match? ", ref_aaspecs[1] == collect(exp_mod.AABASIS_SPECS_1))
println("Order 2 match? ", ref_aaspecs[2] == collect(exp_mod.AABASIS_SPECS_2))

# Get reference A2B matrix
ref_A2B = basis.A2Bmaps[1]
println("\n=== Reference A2B matrix ===")
println("Size: ", size(ref_A2B))
println("Nonzeros: ", nnz(ref_A2B))

# Compare with exported
exp_A2B_size = exp_mod.A2BMAP_1_SIZE
exp_A2B_nnz = length(exp_mod.A2BMAP_1_V)
println("\nExported A2B size: ", exp_A2B_size)
println("Exported A2B nnz: ", exp_A2B_nnz)

# Build dense A2B from sparse components for comparison
exp_A2B_dense = zeros(Float64, exp_A2B_size...)
for k in 1:length(exp_mod.A2BMAP_1_I)
    exp_A2B_dense[exp_mod.A2BMAP_1_I[k], exp_mod.A2BMAP_1_J[k]] = exp_mod.A2BMAP_1_V[k]
end
ref_A2B_dense = Matrix(ref_A2B)

println("\nA2B dense comparison:")
println("ref_A2B_dense[1:5, 1:5] =")
for i in 1:5
    println("  ", ref_A2B_dense[i, 1:5])
end
println("\nexp_A2B_dense[1:5, 1:5] =")
for i in 1:5
    println("  ", exp_A2B_dense[i, 1:5])
end

println("\nA2B matrices match? ", ref_A2B_dense ≈ exp_A2B_dense)

# Now let's evaluate the reference model step by step
# We need to evaluate site energy for atom 1 using reference
println("\n=== Step-by-step reference evaluation ===")

# Get full embeddings for atom 1 using reference model
# First, evaluate Rnl for each neighbor
Rs1, Zs1 = neighbor_data[1]
n_neighbors = length(Rs1)
println("Atom 1 has $n_neighbors neighbors")

# Collect all edges
edges_atom1 = [(𝐫 = R, z0 = AtomsBase.ChemicalSpecies(:Si), z1 = AtomsBase.ChemicalSpecies(:Si)) for R in Rs1]

# Direct evaluation - evaluate rembed and yembed
# The model structure is:
# rembed: EdgeEmbed{TransSelSplines{...}}
# yembed: EdgeEmbed{EmbedDP{...}}
# basis: SparseACEbasis{...}
# readout: SelectLinL{...}

# Try to call the model directly
# ETACE is a Lux model, so we need (model)(input, ps, st)

# Let's check the model's forward signature
println("\nTrying to call ETACE model...")

# Build input for ETACE
# The input should be an ETGraph or similar structure
# Looking at ACEpotentials.ETModels, the model takes graph-based input

# Actually, let's evaluate site energy through the calculator directly
# and compare individual components

# Evaluate site contributions for reference
println("\n=== Computing reference site energies manually ===")

# The WrappedSiteCalculator should have a method to evaluate site energy
# Let's check if we can access it
calc = calc_splined
println("calc type: ", typeof(calc))

# Check the model's call structure - ETACE should compute:
# 1. rembed(edges) -> Rnl for each edge
# 2. yembed(edges) -> Ylm for each edge
# 3. basis(Rnl, Ylm) -> B-basis
# 4. readout(B) -> energy

# For direct comparison, let me print the WB weights
WB_ref = et_ps_splined.readout.W
println("\nReference WB (readout.W):")
println("  Shape: ", size(WB_ref))
println("  WB[1, 1:10, 1] = ", WB_ref[1, 1:10, 1])

WB_exp = exp_mod.WB_1
println("\nExported WB_1[1:10] = ", WB_exp[1:10])

println("\nWB match? ", vec(WB_ref[1,:,1]) ≈ WB_exp)

# Now check Rnl weights (they should be baked into splines, so exported shouldn't have W)
# The splinified model has already incorporated the W matrix into F,G

# Key insight: Reference model applies WB to B-basis
# Exported model applies WB to B-basis
# Both WB match. So issue must be in B-basis computation.

# Let's manually compute B from reference for atom 1
println("\n=== Manual B-basis computation for reference ===")

# To compute B:
# 1. Compute A[k] = sum_j (Rnl[j,n_k] * Ylm[j,l_k]) where (n_k, l_k) = aspec[k]
# 2. Compute AA[k] from products of A
# 3. B = A2B * AA

# But first, we need Rnl and Ylm from reference for each neighbor

# The issue might be: reference evaluates on GRAPH edges with batching
# while exported evaluates individually

# Let's check if there's a pooling/summing step we're missing

# From exported model, compute_embeddings returns (Rnl, Ylm) matrices
# where rows are neighbors, columns are basis functions
println("\n=== Checking embedding dimensions ===")
Rnl_exp, Ylm_exp = exp_mod.compute_embeddings(Rs1, Zs1, 14)
println("Exported Rnl shape: ", size(Rnl_exp))
println("Exported Ylm shape: ", size(Ylm_exp))

# The A-basis should be computed as sum over neighbors
println("\n=== A-basis computation check ===")

# Manual A-basis from exported embeddings
N_A = length(exp_mod.ABASIS_SPEC)
A_manual = zeros(Float64, N_A)
for (iA, (n, l)) in enumerate(exp_mod.ABASIS_SPEC)
    for j in 1:n_neighbors
        A_manual[iA] += Rnl_exp[j, n] * Ylm_exp[j, l]
    end
end
println("Manual A-basis[1:10] = ", A_manual[1:10])
println("Sum(A_manual) = ", sum(A_manual))

# Check what exported model computes for A
_, A_exp = exp_mod.tensor_evaluate(Rnl_exp, Ylm_exp)
println("Exported A-basis[1:10] = ", A_exp[1:10])
println("Sum(A_exp) = ", sum(A_exp))

println("\nA-basis match manual? ", A_manual ≈ A_exp)
