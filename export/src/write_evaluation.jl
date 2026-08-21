# Evaluation functions writing
# Split from export_ace_model.jl for maintainability

function _write_evaluation_functions(io, tensor, NZ, has_pair)
    # Get dimensions for pre-allocation
    nA = length(tensor.abasis)
    nAA = length(tensor.aabasis)

    println(io, """
# ============================================================================
# PRE-ALLOCATED WORK ARRAYS (avoid allocations in hot paths)
# ============================================================================

const MAX_NEIGHBORS = 256  # Maximum number of neighbors per site

# Work arrays for embeddings (indexed as [neighbor, feature])
# Layout matches the abasis inner loop: Rnl[j, ϕ1] with j varying is contiguous
const WORK_Rnl = zeros(Float64, MAX_NEIGHBORS, N_RNL)
const WORK_dRnl = zeros(Float64, MAX_NEIGHBORS, N_RNL)
const WORK_Ylm = zeros(Float64, MAX_NEIGHBORS, N_YLM)
const WORK_dYlm = zeros(SVector{3, Float64}, MAX_NEIGHBORS, N_YLM)
const WORK_rs = zeros(Float64, MAX_NEIGHBORS)
const WORK_rhats = zeros(SVector{3, Float64}, MAX_NEIGHBORS)

# Work arrays for tensor evaluation
const WORK_A = zeros(Float64, $nA)
const WORK_AA = zeros(Float64, $nAA)
const WORK_B = zeros(Float64, N_BASIS)

# Work arrays for pullback
const WORK_∂A = zeros(Float64, $nA)
const WORK_∂AA = zeros(Float64, $nAA)
const WORK_∂Rnl = zeros(Float64, MAX_NEIGHBORS, N_RNL)
const WORK_∂Ylm = zeros(Float64, MAX_NEIGHBORS, N_YLM)
const WORK_∂B = zeros(Float64, N_BASIS)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

# Compute embeddings for all neighbors (values only)
# Uses pre-allocated arrays, returns views
function compute_embeddings(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    nneigh = length(Rs)
    @assert nneigh <= MAX_NEIGHBORS "Too many neighbors: \$nneigh > \$MAX_NEIGHBORS"
    iz0 = z2i(Z0)

    # Get views into pre-allocated arrays
    Rnl = view(WORK_Rnl, 1:nneigh, :)
    Ylm = view(WORK_Ylm, 1:nneigh, :)

    @inbounds for j in 1:nneigh
        r = norm(Rs[j])
        if r > 1e-10
            jz = z2i(Zs[j])
            Rnl_j = evaluate_Rnl(r, iz0, jz)
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = Rnl_j[t]
            end

            # Solid harmonics: evaluate with full R vector (not unit vector!)
            Ylm_j = eval_ylm(Rs[j])
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = Ylm_j[t]
            end
        else
            # Zero out for this neighbor (r too small)
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = 0.0
            end
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = 0.0
            end
        end
    end

    return Rnl, Ylm
end

# Compute embeddings with derivatives for analytic forces
# Uses pre-allocated arrays, returns views
function compute_embeddings_ed(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    nneigh = length(Rs)
    @assert nneigh <= MAX_NEIGHBORS "Too many neighbors: \$nneigh > \$MAX_NEIGHBORS"
    iz0 = z2i(Z0)

    # Get views into pre-allocated arrays
    Rnl = view(WORK_Rnl, 1:nneigh, :)
    dRnl = view(WORK_dRnl, 1:nneigh, :)
    Ylm = view(WORK_Ylm, 1:nneigh, :)
    dYlm = view(WORK_dYlm, 1:nneigh, :)
    rs = view(WORK_rs, 1:nneigh)
    rhats = view(WORK_rhats, 1:nneigh)

    @inbounds for j in 1:nneigh
        r = norm(Rs[j])
        WORK_rs[j] = r
        if r > 1e-10
            rhat = Rs[j] / r
            WORK_rhats[j] = rhat
            jz = z2i(Zs[j])

            # Radial basis with derivative
            Rnl_j, dRnl_j = evaluate_Rnl_d(r, iz0, jz)
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = Rnl_j[t]
                WORK_dRnl[j, t] = dRnl_j[t]
            end

            # Solid harmonics with derivatives (evaluated with full R vector)
            # dYlm returns dR_lm/dR directly (not dY_lm/dr̂)
            Ylm_j, dYlm_j = eval_ylm_ed(Rs[j])
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = Ylm_j[t]
                WORK_dYlm[j, t] = dYlm_j[t]
            end
        else
            # Zero out for this neighbor (r too small)
            WORK_rhats[j] = zero(SVector{3, Float64})
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = 0.0
                WORK_dRnl[j, t] = 0.0
            end
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = 0.0
                WORK_dYlm[j, t] = zero(SVector{3, Float64})
            end
        end
    end

    return Rnl, dRnl, Ylm, dYlm, rs, rhats
end

# Site energy evaluation
function site_energy(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)

    if length(Rs) == 0
        # Return E0 for isolated atom
""")

    # Write E0 lookup
    _emit_species_dispatch(io, NZ, "        ", iz -> "return E0_$iz")

    println(io, """
    end

    # Compute embeddings
    Rnl, Ylm = compute_embeddings(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    B, _ = tensor_evaluate(Rnl, Ylm)

    # Contract with weights
    val = 0.0
""")

    # Write weight contraction
    _emit_species_dispatch_multi(io, NZ, "    ", iz -> ["val = dot(B, WB_$iz)"])

    # Add E0
    println(io, "\n    # Add reference energy")
    _emit_species_dispatch(io, NZ, "    ", iz -> "val += E0_$iz")

    println(io, """

    return val
end

# Site basis evaluation (returns raw basis vector without weight contraction)
function site_basis(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    if length(Rs) == 0
        # Return zeros for isolated atom
        return zeros(Float64, N_BASIS)
    end

    # Compute embeddings
    Rnl, Ylm = compute_embeddings(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    B, _ = tensor_evaluate(Rnl, Ylm)

    return collect(B)
end

# ============================================================================
# MANUAL PULLBACK FUNCTIONS (trim-safe)
# ============================================================================

# Static product with gradient (for SparseSymmProd pullback)
@inline function _static_prod_ed(b::NTuple{1, T}) where {T}
    return b[1], (one(T),)
end

@inline function _static_prod_ed(b::NTuple{2, T}) where {T}
    return b[1] * b[2], (b[2], b[1])
end

@inline function _static_prod_ed(b::NTuple{3, T}) where {T}
    p12 = b[1] * b[2]
    return p12 * b[3], (b[2] * b[3], b[1] * b[3], p12)
end

@inline function _static_prod_ed(b::NTuple{4, T}) where {T}
    p12 = b[1] * b[2]
    p34 = b[3] * b[4]
    return p12 * p34, (b[2] * p34, b[1] * p34, p12 * b[4], p12 * b[3])
end

# Manual pullback through PooledSparseProduct (abasis): ∂A -> (∂Rnl, ∂Ylm)
# This is the key function that replaces ET.pullback for the abasis
@inline function pullback_abasis!(∂Rnl, ∂Ylm, ∂A, Rnl, Ylm)
    nX = size(Rnl, 1)

    @inbounds for (iA, ϕ) in enumerate(ABASIS_SPEC)
        ϕ1, ϕ2 = ϕ  # (Rnl index, Ylm index)
        ∂A_iA = ∂A[iA]
        @simd ivdep for j = 1:nX
            ∂Rnl[j, ϕ1] += ∂A_iA * Ylm[j, ϕ2]
            ∂Ylm[j, ϕ2] += ∂A_iA * Rnl[j, ϕ1]
        end
    end
    return ∂Rnl, ∂Ylm
end
""")

    # Write the aabasis pullback with the correct order
    aabasis = tensor.aabasis
    max_order = length(aabasis.specs)

    println(io, """
# Manual pullback through SparseSymmProd (aabasis): ∂AA -> ∂A
# Note: Accepts any array types to support views
@inline function pullback_aabasis!(∂A, ∂AA, A)
""")

    # Generate pullback code for each order
    for ord in 1:max_order
        spec = aabasis.specs[ord]
        range_start = aabasis.ranges[ord].start
        range_stop = aabasis.ranges[ord].stop

        if isempty(spec)
            continue
        end

        println(io, "    # Order $ord terms (indices $range_start:$range_stop)")
        println(io, "    @inbounds for (i_local, ϕ) in enumerate(AABASIS_SPECS_$ord)")
        println(io, "        i = $(range_start - 1) + i_local")
        println(io, "        ∂AA_i = ∂AA[i]")

        if ord == 1
            println(io, "        ∂A[ϕ[1]] += ∂AA_i")
        elseif ord == 2
            println(io, "        a1, a2 = A[ϕ[1]], A[ϕ[2]]")
            println(io, "        ∂A[ϕ[1]] += ∂AA_i * a2")
            println(io, "        ∂A[ϕ[2]] += ∂AA_i * a1")
        elseif ord == 3
            println(io, "        a1, a2, a3 = A[ϕ[1]], A[ϕ[2]], A[ϕ[3]]")
            println(io, "        ∂A[ϕ[1]] += ∂AA_i * a2 * a3")
            println(io, "        ∂A[ϕ[2]] += ∂AA_i * a1 * a3")
            println(io, "        ∂A[ϕ[3]] += ∂AA_i * a1 * a2")
        elseif ord == 4
            println(io, "        a1, a2, a3, a4 = A[ϕ[1]], A[ϕ[2]], A[ϕ[3]], A[ϕ[4]]")
            println(io, "        ∂A[ϕ[1]] += ∂AA_i * a2 * a3 * a4")
            println(io, "        ∂A[ϕ[2]] += ∂AA_i * a1 * a3 * a4")
            println(io, "        ∂A[ϕ[3]] += ∂AA_i * a1 * a2 * a4")
            println(io, "        ∂A[ϕ[4]] += ∂AA_i * a1 * a2 * a3")
        else
            # General case using _static_prod_ed
            println(io, "        aa = ntuple(t -> A[ϕ[t]], Val($ord))")
            println(io, "        _, gi = _static_prod_ed(aa)")
            println(io, "        for t in 1:$ord")
            println(io, "            ∂A[ϕ[t]] += ∂AA_i * gi[t]")
            println(io, "        end")
        end
        println(io, "    end")
        println(io)
    end

    println(io, "    return ∂A")
    println(io, "end")
    println(io)

    # Write the full tensor pullback
    println(io, """
# ============================================================================
# MANUAL FORWARD EVALUATION FUNCTIONS (trim-safe)
# ============================================================================

# Manual forward pass through PooledSparseProduct (abasis): (Rnl, Ylm) -> A
# This replaces ET.evaluate! for the abasis
@inline function evaluate_abasis!(A, Rnl, Ylm)
    nX = size(Rnl, 1)

    @inbounds for (iA, ϕ) in enumerate(ABASIS_SPEC)
        ϕ1, ϕ2 = ϕ  # (Rnl index, Ylm index)
        acc = 0.0
        @simd ivdep for j = 1:nX
            acc += Rnl[j, ϕ1] * Ylm[j, ϕ2]
        end
        A[iA] = acc
    end
    return A
end""")

    # Get number of A basis functions
    nA = length(tensor.abasis)
    nAA = length(tensor.aabasis)

    println(io, """

# Manual forward pass through SparseSymmProd (aabasis): A -> AA
# This replaces ET.evaluate for the aabasis
# Note: Accepts any array types to support views
@inline function evaluate_aabasis!(AA, A)
""")

    # Generate forward pass code for each order
    aabasis = tensor.aabasis
    max_order = length(aabasis.specs)

    for ord in 1:max_order
        spec = aabasis.specs[ord]
        range_start = aabasis.ranges[ord].start
        range_stop = aabasis.ranges[ord].stop

        if isempty(spec)
            continue
        end

        println(io, "    # Order $ord terms (indices $range_start:$range_stop)")
        println(io, "    @inbounds for (i_local, ϕ) in enumerate(AABASIS_SPECS_$ord)")
        println(io, "        i = $(range_start - 1) + i_local")

        if ord == 1
            println(io, "        AA[i] = A[ϕ[1]]")
        elseif ord == 2
            println(io, "        AA[i] = A[ϕ[1]] * A[ϕ[2]]")
        elseif ord == 3
            println(io, "        AA[i] = A[ϕ[1]] * A[ϕ[2]] * A[ϕ[3]]")
        elseif ord == 4
            println(io, "        AA[i] = A[ϕ[1]] * A[ϕ[2]] * A[ϕ[3]] * A[ϕ[4]]")
        else
            # General case
            println(io, "        AA[i] = prod(A[ϕ[t]] for t in 1:$ord)")
        end
        println(io, "    end")
        println(io)
    end

    println(io, "    return AA")
    println(io, "end")
    println(io)

    # Write full tensor forward evaluation
    println(io, """
# Full manual forward pass through tensor: (Rnl, Ylm) -> B
# This replaces ET.evaluate with a trim-safe implementation
# Uses pre-allocated work arrays to avoid allocations
function tensor_evaluate(Rnl, Ylm)
    # Reset and use pre-allocated arrays
    fill!(WORK_A, 0.0)
    fill!(WORK_AA, 0.0)
    fill!(WORK_B, 0.0)

    # Step 1: A = evaluate(abasis, Rnl, Ylm)
    evaluate_abasis!(WORK_A, Rnl, Ylm)

    # Step 2: AA = evaluate(aabasis, A)
    evaluate_aabasis!(WORK_AA, WORK_A)

    # Step 3: B = A2Bmap * AA (sparse matrix-vector multiplication)
    @inbounds for (idx, I) in enumerate(A2BMAP_1_I)
        J = A2BMAP_1_J[idx]
        V = A2BMAP_1_V[idx]
        WORK_B[I] += V * WORK_AA[J]
    end

    return WORK_B, WORK_A
end

# ============================================================================
# MANUAL PULLBACK FUNCTIONS (trim-safe)
# ============================================================================

# Full manual pullback through tensor: ∂B -> (∂Rnl, ∂Ylm)
# This replaces ET.pullback with a trim-safe implementation
# Uses pre-allocated work arrays to avoid allocations
function tensor_pullback!(∂Rnl, ∂Ylm, ∂B, Rnl, Ylm, A)
    # Reset pre-allocated arrays
    fill!(WORK_∂AA, 0.0)
    fill!(WORK_∂A, 0.0)

    # Step 1: ∂AA = A2Bmap' * ∂B (sparse transpose multiplication)
    @inbounds for (I_idx, I) in enumerate(A2BMAP_1_I)
        J = A2BMAP_1_J[I_idx]
        V = A2BMAP_1_V[I_idx]
        WORK_∂AA[J] += V * ∂B[I]  # Transpose: A2Bmap'[J,I] = A2Bmap[I,J]
    end

    # Step 2: ∂A = pullback_aabasis(∂AA, A)
    pullback_aabasis!(WORK_∂A, WORK_∂AA, A)

    # Step 3: (∂Rnl, ∂Ylm) = pullback_abasis(∂A, Rnl, Ylm)
    pullback_abasis!(∂Rnl, ∂Ylm, WORK_∂A, Rnl, Ylm)

    return ∂Rnl, ∂Ylm
end

# Site energy with ANALYTIC forces using manual pullback (trim-safe)
function site_energy_forces(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)
    nneigh = length(Rs)

    if nneigh == 0
        # Return E0 for isolated atom, no forces
""")

    # Write E0 lookup for forces function
    _emit_species_dispatch(io, NZ, "        ", iz -> "return E0_$iz, SVector{3, Float64}[]")

    println(io, """
    end

    # Compute embeddings with derivatives
    Rnl, dRnl, Ylm, dYlm, rs, rhats = compute_embeddings_ed(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    # Returns both B and A (A needed for pullback)
    B, A = tensor_evaluate(Rnl, Ylm)

    # Contract with weights to get energy
    # Use pre-allocated ∂B array
    fill!(WORK_∂B, 0.0)
    Ei = 0.0
""")

    # Write weight contraction for forces
    _emit_species_dispatch_multi(io, NZ, "    ", iz -> [
        "Ei = dot(B, WB_$iz)",
        "for k in 1:N_BASIS; WORK_∂B[k] = WB_$(iz)[k]; end  # ∂Ei/∂B = WB"
    ])

    println(io, """

    # Backward pass through tensor using MANUAL pullback (trim-safe)
    # Use pre-allocated arrays for gradients
    ∂Rnl = view(WORK_∂Rnl, 1:nneigh, :)
    ∂Ylm = view(WORK_∂Ylm, 1:nneigh, :)
    @inbounds for j in 1:nneigh
        @simd for t in 1:N_RNL; WORK_∂Rnl[j, t] = 0.0; end
        @simd for t in 1:N_YLM; WORK_∂Ylm[j, t] = 0.0; end
    end
    tensor_pullback!(∂Rnl, ∂Ylm, WORK_∂B, Rnl, Ylm, A)

    # Assemble forces: ∂Ei/∂Rⱼ
    forces = Vector{SVector{3, Float64}}(undef, nneigh)
    @inbounds for j in 1:nneigh
        f = zero(SVector{3, Float64})
        r = rs[j]
        if r > 1e-10
            rhat = rhats[j]
            # Contribution from radial basis: ∂Ei/∂Rnl * dRnl/dr * r̂
            for t in 1:N_RNL
                f = f + (∂Rnl[j, t] * dRnl[j, t]) * rhat
            end
            # Contribution from solid harmonics: ∂Ei/∂R_lm * dR_lm/dR
            # (dYlm is dR_lm/dR for solid harmonics - direct gradient, no chain rule needed)
            for t in 1:N_YLM
                f = f + ∂Ylm[j, t] * dYlm[j, t]
            end
        end
        forces[j] = -f  # Force is negative gradient
    end

    # Add reference energy
""")

    # Add E0
    _emit_species_dispatch(io, NZ, "    ", iz -> "Ei += E0_$iz")

    println(io, """

    return Ei, forces
end

# Site energy with forces AND virial stress using manual pullback (trim-safe)
function site_energy_forces_virial(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)
    nneigh = length(Rs)

    # Initialize virial tensor (3x3 symmetric)
    virial = zeros(SMatrix{3, 3, Float64, 9})

    if nneigh == 0
        # Return E0 for isolated atom, no forces/virial
""")

    # Write E0 lookup for virial function
    _emit_species_dispatch(io, NZ, "        ", iz -> "return E0_$iz, SVector{3, Float64}[], virial")

    println(io, """
    end

    # Compute embeddings with derivatives
    Rnl, dRnl, Ylm, dYlm, rs, rhats = compute_embeddings_ed(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    # Returns both B and A (A needed for pullback)
    B, A = tensor_evaluate(Rnl, Ylm)

    # Contract with weights to get energy
    # Use pre-allocated ∂B array
    fill!(WORK_∂B, 0.0)
    Ei = 0.0
""")

    # Write weight contraction for virial
    _emit_species_dispatch_multi(io, NZ, "    ", iz -> [
        "Ei = dot(B, WB_$iz)",
        "for k in 1:N_BASIS; WORK_∂B[k] = WB_$(iz)[k]; end  # ∂Ei/∂B = WB"
    ])

    println(io, """

    # Backward pass through tensor using MANUAL pullback (trim-safe)
    # Use pre-allocated arrays for gradients
    ∂Rnl = view(WORK_∂Rnl, 1:nneigh, :)
    ∂Ylm = view(WORK_∂Ylm, 1:nneigh, :)
    @inbounds for j in 1:nneigh
        @simd for t in 1:N_RNL; WORK_∂Rnl[j, t] = 0.0; end
        @simd for t in 1:N_YLM; WORK_∂Ylm[j, t] = 0.0; end
    end
    tensor_pullback!(∂Rnl, ∂Ylm, WORK_∂B, Rnl, Ylm, A)

    # Assemble forces and virial
    forces = Vector{SVector{3, Float64}}(undef, nneigh)
    @inbounds for j in 1:nneigh
        f = zero(SVector{3, Float64})
        r = rs[j]
        if r > 1e-10
            rhat = rhats[j]
            Rj = Rs[j]
            # Contribution from radial basis
            for t in 1:N_RNL
                df = (∂Rnl[j, t] * dRnl[j, t]) * rhat
                f = f + df
                # Virial: -Rⱼ ⊗ fⱼ (outer product)
                virial = virial - Rj * df'
            end
            # Contribution from solid harmonics: ∂Ei/∂R_lm * dR_lm/dR
            for t in 1:N_YLM
                df = ∂Ylm[j, t] * dYlm[j, t]
                f = f + df
                virial = virial - Rj * df'
            end
        end
        forces[j] = -f
    end

    # Add reference energy
""")

    # Add E0 for virial
    _emit_species_dispatch(io, NZ, "    ", iz -> "Ei += E0_$iz")

    println(io, """

    return Ei, forces, virial
end
""")
end
