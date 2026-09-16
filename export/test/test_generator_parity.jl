#=
Generator-to-generator parity (Tasks 5-7).

WHAT THIS FILE IS FOR.  Tasks 5, 6 and 7 change how the generator EMITS a model, never what
the model is.  The plan's global constraints therefore bind each of those steps to the
previous generator's exported model:

    "any force differing by more than 1e-13 RELATIVE is a bug, not a speed-up"

This file is that gate.  It checks out the generator sources from a named commit
(`EXPORT_REF_SHA`), runs them into a throw-away `Module`, exports the SAME calculator with
the old and the new generator, evaluates both generated files on the same configurations and
compares energy, forces and virial RELATIVELY.  It then re-checks the NEW file against the
Julia calculator at the usual absolute 1e-12, so a pair of generators that agree with each
other but not with the model still fails.

WHY A SEPARATE SHA ARGUMENT AND NOT `HEAD~1`.  `HEAD~1` is only the previous GENERATOR while
the working branch is exactly one commit ahead of it; from the second commit of a task onward
it silently becomes "the generator half-way through this task", which is not the reference the
constraint names.  Always pass the SHA explicitly:

    EXPORT_REF_SHA=3570eb8e julia --project=.. -e 'include("test_generator_parity.jl")'

The default is kept at `HEAD~1` (it is what the brief specifies) but the resolved SHA and its
subject line are printed on every run, and a @warn fires when the default was used.

WHICH CASES.  Both benchmark models (Cantor, 5 species, order 3; TiAl, 2 species, order 4) in
both radial modes, plus a DENSE-`W` model.  The dense case is not decoration: both benchmark
models were fitted with `init_Wradial = :onehot` and neither fit touches `Wnlq`, so the
generator's dense radial-mixing branch has no coverage from them at all.  `dense_model()`
below asserts that the model it builds really does produce `RBASIS_ONEHOT = false`.

TOLERANCES.  Generator-vs-generator is RELATIVE at 1e-13 (energy against |E|, forces against
the largest force in the configuration but never against less than 1 eV/Å, virial against
|V|_inf but never against less than 1 eV).  New-generator-vs-model is the usual
`check_export` at an absolute 1e-12 with the per-atom normalisation described in
check_export.jl.  Nothing here may ever be loosened; a breach is reported with its measured
number.
=#

using Test
using Printf
using Random
using Lux, LuxCore
using AtomsBase, Unitful, StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "check_export.jl"))                       # + cantor fixture
include(joinpath(@__DIR__, "fixtures", "tial_fixture.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))   # the CURRENT generator

const PARITY_REPO  = normpath(joinpath(@__DIR__, "..", ".."))
const PARITY_BUILD = mkpath(joinpath(@__DIR__, "build", "parity"))
const PARITY_TOL   = 1e-13      # generator vs generator, RELATIVE
const EXPORT_TOL   = 1e-12      # generated code vs the Julia calculator, absolute

# Every source file the generator is made of.  `export_ace_model.jl` `include`s the other
# five by relative path, so writing all six into one directory is enough to run an old
# generator unmodified.
const GENERATOR_FILES = ("export_ace_model.jl", "write_radial.jl", "write_evaluation.jl",
                         "write_c_interface.jl", "codegen.jl", "splinify.jl")

const REF_SHA_GIVEN = haskey(ENV, "EXPORT_REF_SHA")
const REF_SHA = get(ENV, "EXPORT_REF_SHA", "HEAD~1")

"""
    generator_module(sha) -> Module

A `Module` containing the generator as it was at `sha`.  The six source files are extracted
with `git show` into a temporary directory (the working tree is never touched, so this is
safe to run with uncommitted changes -- indeed that is the normal case) and
`export_ace_model.jl` is `include`d into a fresh module.  Memoised per SHA.
"""
const _GEN_CACHE = Dict{String, Module}()
function generator_module(sha::AbstractString)
    haskey(_GEN_CACHE, sha) && return _GEN_CACHE[sha]
    tmp = mktempdir(; prefix = "acegen_")
    for f in GENERATOR_FILES
        src = read(`git -C $PARITY_REPO show $sha:export/src/$f`, String)
        write(joinpath(tmp, f), src)
    end
    m = Module(Symbol("Gen_", replace(string(sha), r"[^A-Za-z0-9]" => "_")))
    # `Module(name)` does NOT bring `eval`/`include` with it on Julia 1.12, and
    # export_ace_model.jl's first act is `include("splinify.jl")`.  Bind them explicitly; the
    # relative path then resolves against the *including file* (Base's task-local include
    # stack), i.e. against `tmp`, which is what makes an old generator run unmodified.
    Core.eval(m, :(eval(x) = Core.eval($m, x)))
    Core.eval(m, :(include(p::AbstractString) = Base.include($m, p)))
    Base.include(m, joinpath(tmp, "export_ace_model.jl"))
    _GEN_CACHE[sha] = m
    return m
end

"Export `calc` with the generator from `sha`."
function export_with(sha::AbstractString, calc, file; mode::Symbol, for_library::Bool = false)
    m = generator_module(sha)
    Base.invokelatest(m.export_ace_model, calc, file;
                      for_library = for_library, radial_basis = mode)
    return file
end

"Export `calc` with the generator in the working tree."
function export_current(calc, file; mode::Symbol, for_library::Bool = false)
    Base.invokelatest(export_ace_model, calc, file;
                      for_library = for_library, radial_basis = mode)
    return file
end

"""
    relative_parity(fold, fnew, held, rcut) -> NamedTuple

Worst RELATIVE disagreement between two generated model files over `held`.

Scales: the energy against `|E_old|`; the forces against the largest force magnitude in the
configuration, floored at 1 eV/Å so a configuration that happens to sit at a stationary point
cannot manufacture a huge ratio out of two tiny numbers; the virial against `|V_old|_inf`,
floored at 1 eV for the same reason.  The raw absolute maxima are returned as well, because
"0.0 relative" and "3e-14 absolute on a 1e3 eV virial" are different pieces of evidence and
the report needs both.
"""
function relative_parity(fold, fnew, held, rcut)
    exo, exn = load_exported(fold), load_exported(fnew)
    dE_rel = dF_rel = dV_rel = 0.0
    dE_abs = dF_abs = dV_abs = 0.0
    bitwise = true
    for sys in held
        Eo, Fo, Vo = Base.invokelatest(exported_efv, exo, sys, rcut)
        En, Fn, Vn = Base.invokelatest(exported_efv, exn, sys, rcut)
        fscale = max(maximum(norm.(Fo)), 1.0)
        vscale = max(maximum(abs.(Vo)), 1.0)
        eE = abs(Eo - En);  eF = maximum(norm.(Fo .- Fn));  eV = maximum(abs.(Vo .- Vn))
        dE_abs = max(dE_abs, eE); dF_abs = max(dF_abs, eF); dV_abs = max(dV_abs, eV)
        dE_rel = max(dE_rel, eE / max(abs(Eo), 1.0))
        dF_rel = max(dF_rel, eF / fscale)
        dV_rel = max(dV_rel, eV / vscale)
        bitwise &= (Eo === En) && all(Fo .=== Fn) && all(Vo .=== Vn)
    end
    worst = max(dE_rel, dF_rel, dV_rel)
    return (; worst, dE_rel, dF_rel, dV_rel, dE_abs, dF_abs, dV_abs, bitwise)
end

# --------------------------------------------------------------------------------------
# The DENSE-W case.  Both benchmark models carry init_Wradial = :onehot, so without this the
# generator's dense radial-mixing branch would be exported by nothing in this repository's
# gated set.  NZ = 2, order 3, small basis: it exists to exercise a code path, not to be fast.
# --------------------------------------------------------------------------------------
const DENSE_ELEMENTS = (:Ti, :Al)
const DENSE_RCUT = 5.5

"""
    dense_model() -> (; stacked, held, rcut)

A small two-species model whose radial weight tensor `Wnlq` is genuinely DENSE
(`init_Wradial = :glorot_normal`), together with three rattled BCC cells to check it on.
Construction follows `ms3_model` in test_multispecies.jl, which is the file that established
that `:glorot_normal` produces per-ordered-pair asymmetric, fully populated weights.
"""
function dense_model()
    d = M._default_rin0cuts(DENSE_ELEMENTS)
    NZ = length(DENSE_ELEMENTS)
    rin0cuts = SMatrix{NZ,NZ}([(rin = 0.0, r0 = d[i, j].r0, rcut = DENSE_RCUT)
                               for i in 1:NZ, j in 1:NZ])
    pair_basis = M.ace_learnable_Rnlrzz(; elements = DENSE_ELEMENTS, level = M.TotalDegree(),
                                          max_level = 8, maxl = 0, maxn = 8,
                                          rin0cuts = rin0cuts,
                                          transforms = (:agnesi, 1, 4), envelopes = :poly1sr,
                                          Winit = :glorot_normal)
    model = M.ace_model(; elements = DENSE_ELEMENTS, order = 3, Ytype = :solid,
                          level = M.TotalDegree(), max_level = 7, rin0cuts = rin0cuts,
                          pair_basis = pair_basis,
                          init_WB = :glorot_normal, init_Wpair = :glorot_normal,
                          init_Wradial = :glorot_normal,
                          E0s = Dict(:Ti => -1.1, :Al => -2.2))
    ps, st = Lux.setup(MersenneTwister(11), model)
    stacked = ETM.convert2et_full(model, ps, st)
    return (; stacked, held = dense_configs(3), rcut = DENSE_RCUT)
end

"Rattled 54-atom BCC cells with both species present, so every ordered pair occurs."
function dense_configs(n)
    a = 3.2
    box = [SVector(3a, 0.0, 0.0), SVector(0.0, 3a, 0.0), SVector(0.0, 0.0, 3a)]
    out = []
    for k in 1:n
        rng = MersenneTwister(200 + k)
        pos = SVector{3,Float64}[]; spec = Symbol[]; idx = 0
        for ix in 0:2, iy in 0:2, iz in 0:2, b in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
            idx += 1
            push!(pos, a .* (SVector(ix, iy, iz) .+ SVector(b)) +
                       0.12 * randn(rng, SVector{3,Float64}))
            push!(spec, DENSE_ELEMENTS[mod1(idx + k, 2)])
        end
        push!(out, AtomsBase.periodic_system([e => x * u"Å" for (e, x) in zip(spec, pos)],
                                             box .* u"Å"))
    end
    return out
end

"""
    parity_cases() -> Vector of (name, calc, reference_calc, held, rcut, mode)

`calc` is what gets exported; `reference_calc` is what the NEW export is gated against at
1e-12 (the SPLINIFIED stack for `:hermite_spline`, the fitted stack for `:polynomial` -- the
same split verify_bench_models.jl makes, and for the same reason).
"""
function parity_cases()
    cases = Any[]
    cx = load_cantor_fixture()
    push!(cases, ("cantor_poly", cx.stacked, cx.stacked, cx.held, cx.rcut, :polynomial))
    push!(cases, ("cantor_h50", cantor_spline_stack(cx; Nspl = 50),
                  cantor_spline_stack(cx; Nspl = 50), cx.held, cx.rcut, :hermite_spline))
    tx = load_tial_fixture()
    push!(cases, ("tial_poly", tx.stacked, tx.stacked, tx.held, tx.rcut, :polynomial))
    push!(cases, ("tial_h50", tial_spline_stack(tx; Nspl = 50),
                  tial_spline_stack(tx; Nspl = 50), tx.held, tx.rcut, :hermite_spline))
    dx = dense_model()
    push!(cases, ("dense_poly", dx.stacked, dx.stacked, dx.held, dx.rcut, :polynomial))
    return cases
end

function _resolved_ref()
    sha = strip(read(`git -C $PARITY_REPO rev-parse --short $REF_SHA`, String))
    subj = strip(read(`git -C $PARITY_REPO log -1 --format=%s $REF_SHA`, String))
    return sha, subj
end

@testset "generator parity vs $REF_SHA" verbose = true begin
    ref_sha, ref_subj = _resolved_ref()
    println("\n" * "="^100)
    println("generator parity reference: EXPORT_REF_SHA=$REF_SHA  ->  $ref_sha  \"$ref_subj\"")
    println("relative tolerance $(PARITY_TOL) (energy/forces/virial), export gate $(EXPORT_TOL)")
    println("="^100); flush(stdout)
    REF_SHA_GIVEN || @warn """
        EXPORT_REF_SHA was not set, so the default HEAD~1 ($ref_sha) is being used.  That is
        the previous GENERATOR only while this branch is exactly one commit ahead of it.
        Pass the task's base SHA explicitly."""

    for (name, calc, refcalc, held, rcut, mode) in parity_cases()
        @testset "$name ($mode)" begin
            fold = joinpath(PARITY_BUILD, "$(name)_ref.jl")
            fnew = joinpath(PARITY_BUILD, "$(name)_new.jl")
            export_with(REF_SHA, calc, fold; mode = mode)
            export_current(calc, fnew; mode = mode)
            @printf("[%s] generated source: ref %.2f MB, new %.2f MB (%.2fx)\n", name,
                    filesize(fold) / 2^20, filesize(fnew) / 2^20,
                    filesize(fnew) / max(filesize(fold), 1))
            flush(stdout)

            p = relative_parity(fold, fnew, held, rcut)
            println("[$name] generator parity vs $ref_sha")
            @printf("    relative:  dE %.3e   dF %.3e   dV %.3e   (tol %.0e)\n",
                    p.dE_rel, p.dF_rel, p.dV_rel, PARITY_TOL)
            @printf("    absolute:  dE %.3e eV   dF %.3e eV/Å   dV %.3e eV\n",
                    p.dE_abs, p.dF_abs, p.dV_abs)
            println("    bit-identical on every configuration: $(p.bitwise)")
            flush(stdout)
            @test p.dE_rel <= PARITY_TOL
            @test p.dF_rel <= PARITY_TOL
            @test p.dV_rel <= PARITY_TOL

            # Agreeing with the old generator is not enough: both could be wrong together.
            dE, dF, dV = check_export_report(fnew, refcalc, held, rcut;
                                             label = "$name NEW vs its reference calculator")
            @test dE <= EXPORT_TOL
            @test dF <= EXPORT_TOL
            @test dV <= EXPORT_TOL

            # The dense case exists to cover the dense radial-mixing branch; prove that it
            # actually takes it rather than silently being one-hot like everything else.
            if startswith(name, "dense")
                src = read(fnew, String)
                @test occursin("RBASIS_ONEHOT = false", src)
            elseif mode == :polynomial
                src = read(fnew, String)
                @test occursin("RBASIS_ONEHOT = true", src)
            end
        end
    end
end
