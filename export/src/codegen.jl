# Code generation utilities for trim-safe exports
#
# These functions generate inline code derived from existing package implementations,
# avoiding Val-based dispatch and other patterns that break trim=safe.
#
# The key insight: we use the upstream packages' code generators at EXPORT TIME,
# then emit specialized code that doesn't require those packages at RUNTIME.
#
# Usage:
#   ylm_code = generate_solid_harmonics_code(maxl)
#   # Write to a separate file, then include() in main export
#
# This file used to carry `generate_hermite_spline_code` / `hermite_pair_rows` as well, the
# code generator for the `:hermite_spline` radial mode.  That mode was removed -- see
# `export/bench/FINDINGS_parity.md` §7 -- so what is left is the solid harmonics.

using SpheriCart
using StaticArrays
using Pkg
using Printf

"""
    generate_solid_harmonics_code(maxl::Int; T=Float64, normalisation=:L2)

Generate trim-safe inline solid harmonics code for a specific maxl value.
Uses SpheriCart's internal code generators but produces output that doesn't
require Val-based dispatch at runtime.

Returns a String containing Julia code.
"""
function generate_solid_harmonics_code(maxl::Int; T=Float64, normalisation=:L2)
    n_ylm = (maxl + 1)^2

    io = IOBuffer()

    # Get SpheriCart version for documentation
    sc_version = try
        deps = Pkg.dependencies()
        sc_uuid = Base.PkgId(SpheriCart).uuid
        string(deps[sc_uuid].version)
    catch
        "unknown"
    end

    println(io, """
# ============================================================================
# SOLID HARMONICS (inline, trim-safe)
# ============================================================================
# Generated from SpheriCart v$sc_version internals
# maxl = $maxl, normalisation = $normalisation
#
# This code is derived from SpheriCart's recurrence relations but avoids
# Val-based dispatch for trim=safe compatibility.

const MAXL = $maxl
const N_YLM = $n_ylm
""")

    # Generate value-only code using SpheriCart's internal generator
    value_code = SpheriCart._codegen_Zlm(maxl, T, normalisation)

    println(io, """
# Solid harmonics evaluation (values only)
@inline function eval_ylm(R::SVector{3, TT}) where {TT}
    x, y, z = R[1], R[2], R[3]
""")

    # Emit the generated code (skip the return statement, we'll add our own)
    for expr in value_code[1:end-1]
        println(io, "    ", expr)
    end

    # Add the return with proper type annotation
    z_vars = join(["Z_$i" for i in 1:n_ylm], ", ")
    println(io, "    return SVector{$n_ylm, TT}($z_vars)")
    println(io, "end")
    println(io)

    # Handle L=0 specially (SpheriCart._codegen_Zlm_grads has a bug for L=0)
    if maxl == 0
        # For L=0, Y_0^0 is constant (0.28209479...), so gradient is zero
        println(io, """
# Solid harmonics with gradients (L=0 special case)
# Y_0^0 is a constant, so its gradient is zero
@inline function eval_ylm_ed(R::SVector{3, TT}) where {TT}
    x, y, z = R[1], R[2], R[3]
    # Y_0^0 = 1/(2*sqrt(pi)) = 0.28209479177387814
    Z_1 = TT(0.28209479177387814)
    Z = SVector{1, TT}(Z_1)
    dZ = SVector{1, SVector{3, TT}}(zero(SVector{3, TT}))
    return Z, dZ
end
""")
    else
        # Generate gradient code using SpheriCart's internal generator
        grad_code = SpheriCart._codegen_Zlm_grads(maxl, T, normalisation)

        println(io, """
# Solid harmonics with gradients
@inline function eval_ylm_ed(R::SVector{3, TT}) where {TT}
    x, y, z = R[1], R[2], R[3]
""")

        # Emit the generated code (skip the return statement)
        for expr in grad_code[1:end-1]
            println(io, "    ", expr)
        end

        # Add the return with proper types
        dz_vars = join(["dZ_$i" for i in 1:n_ylm], ", ")
        println(io, "    Z = SVector{$n_ylm, TT}($z_vars)")
        println(io, "    dZ = SVector{$n_ylm, SVector{3, TT}}($dz_vars)")
        println(io, "    return Z, dZ")
        println(io, "end")
    end

    return String(take!(io))
end
