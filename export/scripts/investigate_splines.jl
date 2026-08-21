#!/usr/bin/env julia
# Investigation script to understand P4ML spline structure
# Phase 1 of cubic B-spline code generation plan

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACE
using Lux
using LuxCore
using Random

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

println("="^80)
println("P4ML Spline Structure Investigation")
println("="^80)

# Create a minimal ETACE model
println("\n[1] Creating minimal ETACE model...")

elements = (:Ti, :Al)
order = 2
totaldegree = 6
rcut = 5.5
maxl = 1

println("   Elements: $elements")
println("   Order: $order, Totaldegree: $totaldegree, Maxl: $maxl")

# Create ACE model first
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = order,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = totaldegree,
    maxl = maxl,
    pair_maxn = totaldegree,
    rin0cuts = rin0cuts,
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)
println("   ACE model basis size: $(length(ps.WB[:, 1]))")

# Convert to ETACE
println("\n[2] Converting to ETACE...")
etace = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), etace)

# Copy radial basis parameters
n_species = length(elements)
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
end

# Copy readout parameters
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

println("   ETACE model type: $(typeof(etace))")

# Splinify the model
println("\n[3] Splinifying ETACE model with Nspl=50...")

etace_splined = splinify(etace, et_ps, et_st; Nspl=50)

println("   Splinified model type: $(typeof(etace_splined))")

# Navigate to splined radial basis
println("\n[4] Navigating to splined radial basis...")

rembed_layer = etace_splined.rembed.layer
println("   rembed_layer type: $(typeof(rembed_layer))")

println("\n   rembed_layer fields:")
for fname in fieldnames(typeof(rembed_layer))
    fval = getfield(rembed_layer, fname)
    println("     $fname :: $(typeof(fval))")
end

# Examine the refstate - this should contain the spline data
println("\n[5] Examining refstate (should contain spline coefficients)...")

refstate = rembed_layer.refstate
println("   refstate type: $(typeof(refstate))")
println("\n   refstate fields:")
for fname in fieldnames(typeof(refstate))
    fval = getfield(refstate, fname)
    println("     $fname :: $(typeof(fval))")
    if fname == :F || fname == :G
        println("       Size: $(size(fval))")
        if size(fval, 1) > 0 && size(fval, 2) > 0
            println("       First element: $(fval[1, 1])")
            println("       Element type: $(typeof(fval[1, 1]))")
            if typeof(fval[1, 1]) <: AbstractVector
                println("       Element length: $(length(fval[1, 1]))")
            end
        end
    elseif fname == :x0 || fname == :x1
        println("       Length: $(length(fval))")
        if length(fval) > 0
            println("       First few values: $(fval[1:min(5, end)])")
        end
    end
end

# F and G should contain the spline coefficients
# x0 and x1 should define the knot boundaries
println("\n[6] Analyzing spline structure...")
println("   F matrix: $(size(refstate.F)) - coefficients for function values")
println("   G matrix: $(size(refstate.G)) - coefficients for derivatives")
println("   x0 vector: $(length(refstate.x0)) knot segments - lower bounds")
println("   x1 vector: $(length(refstate.x1)) knot segments - upper bounds")

if size(refstate.F, 1) > 0 && size(refstate.F, 2) > 0
    println("\n[7] Examining first spline (pair index 1)...")
    F_first = refstate.F[1, 1]
    G_first = refstate.G[1, 1]
    println("   F coefficients (first segment): $F_first")
    println("   G coefficients (first segment): $G_first")
    println("   Number of basis functions: $(length(F_first))")

    println("\n   Knot structure:")
    println("     Segment 1: [$(refstate.x0[1]), $(refstate.x1[1])]")
    if length(refstate.x0) > 1
        println("     Segment 2: [$(refstate.x0[2]), $(refstate.x1[2])]")
        println("     ...")
        println("     Segment $(length(refstate.x0)): [$(refstate.x0[end]), $(refstate.x1[end])]")
    end
end

println("\n[8] Understanding evaluation...")
println("   The splines are evaluated using:")
println("     - trans: Applies Agnesi transform r -> y")
println("     - F, G: Coefficient matrices for each segment and species pair")
println("     - x0, x1: Knot boundaries in transformed space")
println("     - envelope: Cutoff function")
println("     - selector: Chooses correct spline based on (Zi, Zj)")

println("\n" * "="^80)
println("Investigation complete!")
println("="^80)
println("\nKey findings will inform B-spline extraction implementation.")
