# Build stamp: the one definition of a generated model file's identity.
#
# Included by BOTH export/src/export_ace_model.jl (which writes the stamp) and
# export/test/runtests.jl (which recomputes it to check that a compiled `.so` was built from
# the `.jl` beside it).  Kept in its own file precisely so that there is one definition:
# recomputing an identity from a second reading of the same rule, in a second file, is how a
# provenance check silently stops checking anything.

"""
    BUILD_STAMP_MARKER

The line that separates the hashed body of a generated model file from its stamp.  Both the
generator and `export_build_id` split on it, so it is written once, here.
"""
const BUILD_STAMP_MARKER = "# ===== ACE EXPORT BUILD STAMP: everything ABOVE this line is hashed ====="

"""
    export_build_id(source) -> UInt64

The build id of a generated model file: a 64-bit FNV-1a hash of everything that precedes
`BUILD_STAMP_MARKER`.

`source` may be a path or the file's contents.  A compiled library reports the same number
from `ace_build_id()`, so comparing the two answers "was this `.so` compiled from this `.jl`?"
-- a question that otherwise has no answer at all, since nothing about a `.so` records the
source it came from.

FNV-1a rather than sha256 or `Base.hash`: this is a provenance check, not a security or a
dedup mechanism, and it must be reproducible from two places (here and the library) without
adding a dependency to `export/Project.toml` or relying on `Base.hash`, whose value is not
stable across Julia versions.

Returns `zero(UInt64)` for a file with no stamp (a model exported before this existed), which
the caller must treat as "unknown", never as "matches".
"""
function export_build_id(source::AbstractString)
    text = isfile(source) ? read(source, String) : source
    i = findfirst(BUILD_STAMP_MARKER, text)
    i === nothing && return zero(UInt64)
    return _fnv1a64(codeunits(text[1:prevind(text, first(i))]))
end

function _fnv1a64(bytes)
    h = 0xcbf29ce484222325
    for b in bytes
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end

