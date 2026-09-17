#!/usr/bin/env julia
#=
profile_tensor_step.jl [cantor|tial ...]  -- the PER-PHASE cost of one site, for two exported
model files of the same model, so that a change to the tensor step can be attributed.

    taskset -c 31 julia --project=export export/bench/profile_tensor_step.jl cantor tial

THIS IS THE ARTEFACT BEHIND `export/bench/README.md`'s "Why the DAG loses on Cantor and would
win on TiAl" TABLE.  Task 7 published that table from a script that was not committed and left
no log, which contradicts the norm stated in the same file -- that quoting a number means
running the tool and pasting what it prints.  This is that tool.

WHAT IT MEASURES, AND WHAT IT IS NOT.  The three phases of `site_energy_forces_virial!`, timed
individually, once per site, in the order the kernel runs them, plus the whole-site call on the
same site:

    _embed_val!        pass 1: per-neighbour radial + harmonics + A accumulation
    _energy_and_∂A!    the TENSOR STEP (the only phase `aa_products` changes)
    _forces_from_∂A!   pass 2: per-neighbour derivatives, forces, virial
    site_energy_forces_virial!   all three, as the C entry points call it

It is a PROFILE, not a protocol timing row.  It runs in Julia rather than through the compiled
library and LAMMPS, it is not the `bench_parity.sh` protocol, and its whole-site figure is
systematically kinder to a large-working-set tensor step than the real thing: it calls the
tensor step immediately before the whole-site call, so the tensor step's buffers are warm.
Task 7 measured 1.14x here where the LAMMPS row said 1.32x. **Use `bench_parity.sh` for any
headline number; use this only to attribute one.** The attribution it supports is robust in a
way the absolute numbers are not: passes 1 and 2 are BYTE-IDENTICAL code in the two model
files being compared (check it with `diff`), so any difference in their timings is a cache or
layout effect and not a code change.

The statistic is the MINIMUM over `--reps` sweeps of the per-site mean, which is the right
statistic for "how long does this code take" on a shared host -- the minimum is the run least
disturbed by other work. Spread across sweeps is printed so a contended run is visible.

Inputs: `bench_parity/<tag>_model.jl` for each of the two variants named by `--a` / `--b`
(default `_b2` and `_b3`), which `verify_bench_models.jl` writes.
=#
using Printf, StaticArrays, LinearAlgebra

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "export", "test", "check_export.jl"))          # + cantor fixture
include(joinpath(REPO, "export", "test", "fixtures", "tial_fixture.jl"))

const ARGS_ = isempty(ARGS) ? ["cantor", "tial"] : ARGS
const MODELS = filter(a -> !startswith(a, "--"), ARGS_)
_opt(name, dflt) = (i = findfirst(a -> startswith(a, "--$name="), ARGS_);
                    i === nothing ? dflt : split(ARGS_[i], "=", limit = 2)[2])
const SUF_A = _opt("a", "_b2")
const SUF_B = _opt("b", "_b3")
const NSITES = parse(Int, _opt("sites", "200"))
const REPS = parse(Int, _opt("reps", "9"))

function phases(model::AbstractString)
    fx = model == "cantor" ? load_cantor_fixture() : load_tial_fixture()
    res = Dict{String, NTuple{4, Float64}}()
    spreads = Dict{String, Float64}()
    for (nm, suf) in (("A", SUF_A), ("B", SUF_B))
        file = joinpath(REPO, "bench_parity", "$(model)_poly$(suf)_model.jl")
        isfile(file) || error("no such model file: $file (run verify_bench_models.jl first)")
        m = load_exported(file)
        sites = Base.invokelatest(site_sets, fx.held[1], fx.rcut)
        ns = min(NSITES, length(sites))
        ws = Base.invokelatest(m.new_workspace)
        F = Vector{SVector{3, Float64}}(undef, maximum(length(s[1]) for s in sites))
        best = (Inf, Inf, Inf, Inf); totals = Float64[]
        for rep in 1:REPS
            a = b = c = d = 0.0
            for (Rs, Zs, Z0, _) in sites[1:ns]
                iz0 = Base.invokelatest(m.z2i, Int(Z0))
                Fv = view(F, 1:length(Rs))
                t = time_ns(); Base.invokelatest(m._embed_val!, ws, Rs, Zs, iz0); a += time_ns() - t
                t = time_ns(); Base.invokelatest(m._energy_and_∂A!, ws, iz0);      b += time_ns() - t
                t = time_ns(); Base.invokelatest(m._forces_from_∂A!, Fv, ws, Rs, Zs, iz0, true); c += time_ns() - t
                t = time_ns(); Base.invokelatest(m.site_energy_forces_virial!, ws, Rs, Zs, Int(Z0), Fv); d += time_ns() - t
            end
            v = (a / ns / 1000, b / ns / 1000, c / ns / 1000, d / ns / 1000)
            sum(v) < sum(best) && (best = v)
            push!(totals, sum(v))
        end
        res[nm] = best
        # Spread over sweeps 2..n.  Sweep 1 is EXCLUDED because it carries this module's
        # first-call compilation -- it runs ~500x the others, and including it made the
        # printed spread a five-digit percentage that told the reader nothing.
        rest = sort(totals[2:end])
        spreads[nm] = length(rest) < 2 ? 0.0 : rest[end] / rest[1] - 1
    end
    a, b = res["A"], res["B"]
    @printf("\n%s   (%s vs %s, %d sites, best of %d sweeps; sweep spread (excl. sweep 1) %.1f %% / %.1f %%)\n",
            uppercase(model), SUF_A, SUF_B, NSITES, REPS,
            100 * spreads["A"], 100 * spreads["B"])
    @printf("  %-18s %10s %10s %8s\n", "phase", SUF_A * " µs", SUF_B * " µs", "B/A")
    for (i, nm) in enumerate(("embed (pass 1)", "tensor step", "forces (pass 2)", "WHOLE SITE call"))
        @printf("  %-18s %10.2f %10.2f %8.2f\n", nm, a[i], b[i], b[i] / a[i])
    end
end

println("profile_tensor_step.jl -- PROFILE, not a protocol timing row (see the header).")
println("loadavg: ", strip(read("/proc/loadavg", String)))
for m in MODELS
    phases(m)
end
