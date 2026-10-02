# Species-pair indexing and the scatter-literal helper, shared by every per-pair table writer.
#
# HISTORY, because the file was renamed and a reader may arrive here from an older commit.
# This was `export/src/splinify.jl` and it additionally held `HermiteSplineData` and
# `extract_hermite_spline_data`, the export-time half of the `:hermite_spline` radial mode.
# That mode was REMOVED (see `export/bench/FINDINGS_parity.md` §7): it was slower than the
# exact `:polynomial` mode on both reference models, approximate by construction, and refused
# outright for per-pair cutoffs.  What survived the removal is what was never about splines at
# all -- the ordered/symmetric pair-index conversion and `_scatter_expr` -- so the file is
# named for what it does.
#
# `ACEpotentials.ETModels.splinify` itself is untouched; it is only its EXPORT path that is
# gone.  A splinified model can no longer be exported at all, and `export_ace_model` says so.

using StaticArrays
using LinearAlgebra: norm
using Printf
using EquivariantTensors
const ET = EquivariantTensors
using DecoratedParticles
using AtomsBase: ChemicalSpecies

# ----------------------------------------------------------------------------------------
# Species-pair indexing -- the export-time half of the single runtime convention.
#
# The generated code knows exactly ONE pair index, `pair_idx(iz, jz) = (iz-1)*NZ + jz`
# (emitted by `_write_species`), which is `ET.catcat2idx`: the index of the SelectLinL
# weights.  Whatever the model stores per SYMMETRIC pair -- the Agnesi transform parameters,
# `ET.catcat2idx_sym` -- is expanded into that ordered layout HERE, at export time, so the
# runtime never needs a second mapping.
#
# `_ordered_pairs(NZ)` enumerates the ordered pairs in table order, and
# `_sym_pair_index(iz, jz, NZ)` is `ET.symidx` (EquivariantTensors utils/selector.jl:62-65),
# i.e. the position of `(min, max)` in the `for i = 1:NZ, j = i:NZ` order that
# `ETModels._convert_agnesi` fills.  Both are used by every per-pair table writer in
# write_radial.jl -- they live HERE rather than there because `export_ace_model.jl` includes
# this file before write_radial.jl and because the conversion is a property of the model's
# storage layout, not of any one writer.
# ----------------------------------------------------------------------------------------

"Ordered species pairs `(k, iz, jz)` with `k = (iz-1)*NZ + jz`, in table order."
_ordered_pairs(NZ::Int) = [((iz - 1) * NZ + jz, iz, jz) for iz in 1:NZ for jz in 1:NZ]

"Position of the symmetric pair {iz, jz} in the `for i = 1:NZ, j = i:NZ` storage (ET.symidx)."
@inline function _sym_pair_index(iz::Int, jz::Int, NZ::Int)
    i, j = min(iz, jz), max(iz, jz)
    return (i - 1) * NZ - (i - 1) * (i - 2) ÷ 2 + (j - i + 1)
end

"""
    _scatter_expr(rows, entries, n_rnl) -> String

The body of a length-`n_rnl` `SVector` literal in which row `rows[i]` carries `entries[i]` and
every other row carries `zero(T)`.

Emitting the scatter as a STATIC tuple, rather than as a loop over a runtime index vector,
is what makes the generated mixing cheap: `rows` is known at export time, so LLVM sees a
vector of compile-time zeros with a handful of inserted values and emits a `zeroinitializer`
plus `length(rows)` scalar stores, instead of `n_rnl` runtime-indexed stores into an `MVector`.
"""
function _scatter_expr(rows::AbstractVector{Int}, entries::AbstractVector{String}, n_rnl::Int)
    @assert length(rows) == length(entries)
    slot = Dict(t => e for (t, e) in zip(rows, entries))
    parts = [get(slot, t, "zero(T)") for t = 1:n_rnl]
    # 8 per line: the generated file is read by humans when a gate fails.
    lines = String[]
    for i = 1:8:n_rnl
        push!(lines, "        " * join(parts[i:min(i + 7, n_rnl)], ", "))
    end
    return join(lines, ",\n")
end

