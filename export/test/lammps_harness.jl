#=
Shared harness for every test that runs an *external* process against the compiled ACE
library: the LAMMPS groups (`test_lammps.jl`, `test_mpi.jl`) and the Python group
(`test_python.jl`).

It exists because three separate copies of "build LD_LIBRARY_PATH, find lmp" had drifted
apart and two of them were wrong:

  * K1 -- the Python group put only `<julia>/lib` on LD_LIBRARY_PATH.  The libstdc++ the
    juliac-compiled library needs (GLIBCXX_3.4.33) lives in `<julia>/lib/julia`, one level
    down, so 8 of the 13 Python tests errored with a GLIBCXX_3.4.30 message.  Both
    directories are now always added, in that order.

  * K2 -- the LAMMPS groups accepted the first `lmp` on PATH without ever checking that it
    RUNS.  On a host where `which lmp` is a wrapper around a binary linked against an absent
    `libmpi.so.12`, that produced 1 failure + 4 errors that looked like plugin faults.
    `find_lammps_exe` now PROBES each candidate (`lmp -h`, exit status 0) with the very
    environment the tests will use, reports why each rejected candidate was rejected, and
    only then returns one.  If no candidate runs, the caller gets `""` and must skip LOUDLY.

Public API (used by test_lammps.jl, test_mpi.jl, test_python.jl):

    julia_runtime_lib_dirs()                     -> Vector{String}
    ace_runtime_env(extra_dirs...)               -> Dict{String,String}   (a copy of ENV)
    find_lammps_exe(env; extra_candidates)       -> String                ("" if none runs)
    read_lammps_data(fn, elements)               -> AbstractSystem
    read_lammps_dump(fn)                         -> (; ids, types, X, F)

Environment overrides, for hosts whose runtime libraries are not in any standard place:

    ACE_LMP                    absolute path to the LAMMPS executable to use (tried first)
    ACE_MPIRUN                 absolute path to the mpirun to use (tried first)
    ACE_TEST_LD_LIBRARY_PATH   extra colon-separated directories, prepended to LD_LIBRARY_PATH

NOT REPRODUCIBLE, ON THE FAILURE PATH ONLY: when `_probe_exe` heals a missing shared object
it takes the FIRST match in `readdir` order under `_LIB_SEARCH_GLOBS`.  A host with the same
`libfoo.so.N` under two EasyBuild modules can therefore produce a different
`LD_LIBRARY_PATH` on two runs.  Every candidate and every rejection is printed, so the
resolved path is always in the log, and `ACE_TEST_LD_LIBRARY_PATH` pins it when it matters.
This is deliberately not made deterministic: the search only ever runs on a host where the
alternative is not running the group at all.
=#

using AtomsBase: periodic_system
using StaticArrays: SVector
using Unitful: @u_str

# ---------------------------------------------------------------------------------------
# runtime library path
# ---------------------------------------------------------------------------------------

"""
    julia_runtime_lib_dirs() -> Vector{String}

`<julia>/lib` **and** `<julia>/lib/julia`.  A juliac-compiled ACE library links against
`libjulia.so` (in the first) and against Julia's own bundled `libstdc++.so.6` (in the
second).  Omitting the second is K1: the system libstdc++ on an EL9 host provides only
GLIBCXX_3.4.30, while code compiled by Julia 1.12 needs GLIBCXX_3.4.33.
"""
function julia_runtime_lib_dirs()
    lib = normpath(joinpath(Sys.BINDIR, "..", "lib"))
    return [lib, joinpath(lib, "julia")]
end

"""
    gcc_runtime_lib_dir() -> String

A GCCcore `lib64` holding a newer `libstdc++.so.6` than the system one, if this host is an
EasyBuild cluster; `""` otherwise.  Only used for the C++ side (LAMMPS and the plugin).
"""
function gcc_runtime_lib_dir()
    for v in ["14.3.0", "13.3.0", "13.2.0", "12.3.0", "12.2.0", "11.3.0"]
        p = "/software/easybuild/software/GCCcore/$v/lib64"
        _isdir(p) && _isfile(joinpath(p, "libstdc++.so.6")) && return p
    end
    return ""
end

"""
    ace_runtime_env(extra_dirs...) -> Dict{String,String}

A copy of `ENV` whose `LD_LIBRARY_PATH` is

    \$ACE_TEST_LD_LIBRARY_PATH : <gcc lib64> : <julia>/lib : <julia>/lib/julia :
    <extra_dirs...> : \$CONDA_PREFIX/lib : <inherited LD_LIBRARY_PATH>

Duplicates and empty entries are dropped, order preserved.  Pass the directory holding
`libace_*.so` and the LAMMPS build directory as `extra_dirs`.
"""
function ace_runtime_env(extra_dirs::AbstractString...)
    dirs = String[]
    append!(dirs, split(get(ENV, "ACE_TEST_LD_LIBRARY_PATH", ""), ':'; keepempty = false))
    push!(dirs, gcc_runtime_lib_dir())
    append!(dirs, julia_runtime_lib_dirs())
    append!(dirs, collect(extra_dirs))
    haskey(ENV, "CONDA_PREFIX") && push!(dirs, joinpath(ENV["CONDA_PREFIX"], "lib"))
    append!(dirs, split(get(ENV, "LD_LIBRARY_PATH", ""), ':'; keepempty = false))

    seen = Set{String}()
    keep = String[]
    for d in dirs
        isempty(d) && continue
        d in seen && continue
        push!(seen, d); push!(keep, String(d))
    end
    env = copy(ENV)
    env["LD_LIBRARY_PATH"] = join(keep, ":")
    return env
end

# ---------------------------------------------------------------------------------------
# LAMMPS executable discovery -- K2
# ---------------------------------------------------------------------------------------

# Roots searched for a shared library the loader says is missing.  Only consulted on the
# failure path, so the cost is paid only by a host that would otherwise skip the group.
const _LIB_SEARCH_GLOBS = [
    "/software/easybuild/software/*/*/lib64",
    "/software/easybuild/software/*/*/lib",
    joinpath(homedir(), "miniconda3", "envs", "*", "lib"),
    joinpath(homedir(), "miniconda3", "lib"),
    joinpath(homedir(), "miniforge3", "envs", "*", "lib"),
]

# A shared cluster's /software tree contains directories this user may not stat (EACCES), so
# every filesystem probe below has to be total.
_isdir(p) = try isdir(p) catch; false end
_readdir(p) = try readdir(p) catch; String[] end
_isfile(p) = try isfile(p) catch; false end

"Directories matching `pattern`, where `*` matches within one path component."
function _glob_dirs(pattern)
    parts = splitpath(pattern)
    cur = [parts[1]]
    for p in parts[2:end]
        nxt = String[]
        if occursin('*', p)
            rx = Regex("^" * replace(p, "." => "\\.", "*" => "[^/]*") * "\$")
            for c in cur, e in _readdir(c)
                occursin(rx, e) && _isdir(joinpath(c, e)) && push!(nxt, joinpath(c, e))
            end
        else
            for c in cur
                _isdir(joinpath(c, p)) && push!(nxt, joinpath(c, p))
            end
        end
        cur = nxt
    end
    return filter(_isdir, cur)
end

"Directory under `_LIB_SEARCH_GLOBS` containing `libname`, or `\"\"`."
function _find_missing_lib_dir(libname)
    for pat in _LIB_SEARCH_GLOBS, d in _glob_dirs(pat)
        _isfile(joinpath(d, libname)) && return d
    end
    return ""
end

"""
    _probe_exe(exe, env; max_fixups = 4) -> (ok::Bool, env, why::String)

Run `exe -h` under `env`.  If the dynamic loader reports a missing shared object, look the
object up under `_LIB_SEARCH_GLOBS`, prepend the directory that holds it to
`LD_LIBRARY_PATH` and retry (at most `max_fixups` times).  Returns the possibly-augmented
environment so the caller uses exactly the one that was proven to work.
"""
function _probe_exe(exe, env; max_fixups = 4)
    env = copy(env)
    for _ in 0:max_fixups
        out = IOBuffer()
        ok = try
            success(pipeline(setenv(`$exe -h`, env); stdout = devnull, stderr = out))
        catch
            false
        end
        ok && return (true, env, "")
        msg = String(take!(out))
        m = match(r"error while loading shared libraries:\s*([^\s:]+)", msg)
        m === nothing && return (false, env, isempty(strip(msg)) ? "does not run" : strip(msg))
        d = _find_missing_lib_dir(m.captures[1])
        isempty(d) && return (false, env, "missing $(m.captures[1]) (not found on this host)")
        env["LD_LIBRARY_PATH"] = d * ":" * get(env, "LD_LIBRARY_PATH", "")
    end
    return (false, env, "too many missing shared libraries")
end

"""
    find_lammps_exe(env; extra_candidates = String[]) -> (exe::String, env)

First candidate that actually RUNS, together with the environment under which it ran.
Candidates, in order: `\$ACE_LMP`, `extra_candidates`, `\$LAMMPS_SRC/../build/lmp`,
`which lmp`, `~/lammps/*/build/lmp`.

Returns `("", env)` when none runs, after printing one line per rejected candidate saying
why.  A caller that gets `""` **must** make the skip loud (a named SKIPPED testset with a
non-zero broken count) -- see `check_cantor_fixture_available` in runtests.jl for the
rationale.
"""
function find_lammps_exe(env; extra_candidates::Vector{String} = String[])
    cands = String[]
    haskey(ENV, "ACE_LMP") && push!(cands, ENV["ACE_LMP"])
    append!(cands, extra_candidates)
    src = get(ENV, "LAMMPS_SRC", "")
    !isempty(src) && isdir(src) && push!(cands, joinpath(dirname(src), "build", "lmp"))
    try
        w = strip(read(`which lmp`, String))
        !isempty(w) && push!(cands, String(w))
    catch
    end
    append!(cands, [joinpath(d, "lmp") for d in _glob_dirs(joinpath(homedir(), "lammps", "*", "build"))])

    seen = Set{String}()
    for c in cands
        (isempty(c) || c in seen) && continue
        push!(seen, c)
        if !isfile(c)
            @info "  lmp candidate rejected: $c (no such file)"
            continue
        end
        ok, env2, why = _probe_exe(c, env)
        if ok
            return (c, env2)
        end
        @info "  lmp candidate rejected: $c ($why)"
    end
    return ("", env)
end

"""
    find_mpirun(lmp_exe, env) -> String

The `mpirun` that belongs to the *same* MPI installation the LAMMPS executable is linked
against, or `""`.  Launching `lmp` under a foreign `mpirun` either fails to start or -- much
worse for a 1-rank-vs-2-rank parity test -- silently gives every rank `MPI_COMM_WORLD` of
size 1, so that the "2 rank" run is really two independent serial runs and the comparison
proves nothing.  The MPI prefix is read out of `ldd <lmp>`: `<prefix>/lib/libmpi.so.N` maps
to `<prefix>/bin/mpirun`.  `\$ACE_MPIRUN` overrides; `which mpirun` is consulted only
when `ldd <lmp>` named no MPI prefix at all.
"""
function find_mpirun(lmp_exe, env)
    cands = String[]
    haskey(ENV, "ACE_MPIRUN") && push!(cands, ENV["ACE_MPIRUN"])
    if !isempty(lmp_exe)
        out = try
            read(setenv(`ldd $lmp_exe`, env), String)
        catch
            ""
        end
        for l in split(out, '\n')
            m = match(r"=>\s*(\S*/)lib(?:64)?/libmpi\.so", l)
            m === nothing && continue
            push!(cands, joinpath(m.captures[1], "bin", "mpirun"))
        end
    end
    # `which mpirun` is consulted ONLY when `ldd` named no MPI prefix.  A PATH mpirun from a
    # different MPI than the one `lmp` is linked against does not usually fail loudly: it
    # launches N processes each with an `MPI_COMM_WORLD` of size 1, so a "2 rank" run is
    # really two independent serial runs, and every rank-parity comparison built on it passes
    # while proving nothing.  Preferring the `ldd`-derived prefix removes that possibility at
    # the source; `run_two_ranks.sh` and `test_mpi.jl` additionally assert the rank count
    # LAMMPS itself reports, because `$ACE_MPIRUN` can still override this.
    if isempty(cands) || (haskey(ENV, "ACE_MPIRUN") && length(cands) == 1)
        try
            w = strip(read(`which mpirun`, String))
            !isempty(w) && push!(cands, String(w))
        catch
        end
    end
    for c in cands
        isfile(c) || continue
        ok = try
            success(pipeline(setenv(`$c --version`, env); stdout = devnull, stderr = devnull))
        catch
            false
        end
        ok && return c
        @info "  mpirun candidate rejected: $c (does not run)"
    end
    return ""
end

# ---------------------------------------------------------------------------------------
# LAMMPS file readers -- the two formats the tests exchange geometry through
# ---------------------------------------------------------------------------------------

"""
    read_lammps_data(fn, elements) -> AbstractSystem

Read a LAMMPS `data` file (as written by `write_data`) into an AtomsBase system, mapping
atom type `t` to `elements[t]`.  Handles the triclinic `xy xz yz` line when present.
Generalisation of `read_cantor_lammps_data` (fixtures/cantor_fixture.jl), which is the
verbatim copy of `verify_cantor/chain_cantor.jl:99-111`; the box convention (cell vectors as
`a = (xhi-xlo, 0, 0)`, `b = (xy, yhi-ylo, 0)`, `c = (xz, yz, zhi-zlo)`) is that script's and
must not be changed -- it is what the 17-digit references were produced with.

Atoms are returned in ascending LAMMPS atom-id order, so index `i` here is atom id `i` in
any dump written with `dump_modify sort id`.
"""
function read_lammps_data(fn, elements)
    L = readlines(fn)
    g(pat) = parse.(Float64, split(L[findfirst(l -> occursin(pat, l), L)])[1:end-length(split(pat))])
    xlo, xhi = g("xlo xhi"); ylo, yhi = g("ylo yhi"); zlo, zhi = g("zlo zhi")
    xy, xz, yz = any(l -> occursin("xy xz yz", l), L) ? Tuple(g("xy xz yz")) : (0.0, 0.0, 0.0)
    box = [SVector(xhi - xlo, 0.0, 0.0), SVector(xy, yhi - ylo, 0.0), SVector(xz, yz, zhi - zlo)]
    ia = findfirst(l -> startswith(l, "Atoms"), L)
    ia === nothing && error("read_lammps_data: no `Atoms` section in $fn")
    rows = Tuple{Int,Symbol,SVector{3,Float64}}[]
    for l in L[ia+2:end]
        isempty(strip(l)) && !isempty(rows) && break
        isempty(strip(l)) && continue
        w = split(l)
        length(w) < 5 && break
        push!(rows, (parse(Int, w[1]), elements[parse(Int, w[2])],
                     SVector(parse.(Float64, w[3:5])...)))
    end
    sort!(rows, by = r -> r[1])
    return periodic_system([r[2] => r[3] * u"Å" for r in rows], box .* u"Å")
end

"""
    read_lammps_dump(fn) -> (; ids, types, X, F)

Read a LAMMPS custom dump whose `ITEM: ATOMS` header names its columns.  `X` and `F` are
`Vector{SVector{3,Float64}}` indexed by **atom id** (so they line up with
`read_lammps_data`); a field that is not dumped comes back as an empty vector.
"""
function read_lammps_dump(fn)
    L = readlines(fn)
    n = parse(Int, strip(L[4]))
    ih = findfirst(l -> startswith(l, "ITEM: ATOMS"), L)
    ih === nothing && error("read_lammps_dump: no `ITEM: ATOMS` line in $fn")
    cols = split(L[ih])[3:end]
    col(name) = findfirst(==(name), cols)
    iid, ity, ix, ifx = col("id"), col("type"), col("x"), col("fx")
    iid === nothing && error("read_lammps_dump: dump has no `id` column")
    ids = zeros(Int, n); types = zeros(Int, n)
    X = ix === nothing ? SVector{3,Float64}[] : Vector{SVector{3,Float64}}(undef, n)
    F = ifx === nothing ? SVector{3,Float64}[] : Vector{SVector{3,Float64}}(undef, n)
    for k in 1:n
        w = split(L[ih+k])
        id = parse(Int, w[iid])
        ids[id] = id
        ity === nothing || (types[id] = parse(Int, w[ity]))
        ix  === nothing || (X[id] = SVector(parse.(Float64, w[ix:ix+2])...))
        ifx === nothing || (F[id] = SVector(parse.(Float64, w[ifx:ifx+2])...))
    end
    return (; ids, types, X, F)
end
