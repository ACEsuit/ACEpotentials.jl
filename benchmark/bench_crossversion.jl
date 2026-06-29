# Cross-version force benchmark driver.
#
# Compares classic `ace1_model` force evaluation between a pinned PREVIOUS
# release of ACEpotentials and the CURRENT checkout, each in its own isolated
# environment, on identical Si systems. This measures release-vs-release drift
# in the default (analytic) force path.
#
# Run:  julia --project=. benchmark/bench_crossversion.jl [old_version]
# (default old_version = 0.9.1, the last release before the ET-backend series)

using Pkg, Printf

const OLD_VERSION = isempty(ARGS) ? "0.9.1" : ARGS[1]
const REPO  = dirname(@__DIR__)
const WORKER = joinpath(@__DIR__, "_crossversion_worker.jl")
const SCRATCH = mktempdir()

worker_deps() = ["AtomsBuilder", "AtomsCalculators", "Unitful", "BenchmarkTools"]

"Set up an isolated env with ACEpotentials pinned to `version` (or `path` for dev)."
function setup_env(dir; version = nothing, path = nothing)
   Pkg.activate(dir)
   if path !== nothing
      Pkg.develop(path = path)
   else
      Pkg.add(Pkg.PackageSpec(name = "ACEpotentials", version = version))
   end
   Pkg.add(worker_deps())
   Pkg.instantiate()
end

"Run the worker in `env` and parse `natoms => ms`."
function run_worker(env, label)
   out = read(`$(Base.julia_cmd()) --project=$env --threads=1 $WORKER $label`, String)
   print(out)
   res = Dict{Int,Float64}()
   for ln in split(out, '\n')
      (isempty(ln) || startswith(ln, "#")) && continue
      parts = split(strip(ln))
      length(parts) == 2 || continue
      res[parse(Int, parts[1])] = parse(Float64, parts[2])
   end
   return res
end

println("Cross-version force benchmark: old=$OLD_VERSION vs current dev\n")

old_env = joinpath(SCRATCH, "old")
new_env = joinpath(SCRATCH, "new")
mkpath(old_env); mkpath(new_env)

@info "Setting up OLD env (ACEpotentials@$OLD_VERSION)…"
setup_env(old_env; version = OLD_VERSION)
@info "Setting up NEW env (current dev checkout)…"
setup_env(new_env; path = REPO)

@info "Benchmarking OLD…"
old = run_worker(old_env, "old-$OLD_VERSION")
@info "Benchmarking NEW…"
new = run_worker(new_env, "new-dev")

println("\n", "="^60)
println("| Atoms | old $OLD_VERSION (ms) | new dev (ms) | new/old |")
println("|-------|--------------|--------------|---------|")
for nat in sort(collect(keys(old)))
   haskey(new, nat) || continue
   @printf("| %5d | %12.3f | %12.3f | %7.2f |\n", nat, old[nat], new[nat], new[nat]/old[nat])
end
println("\n(new/old > 1 ⇒ current release is SLOWER than $OLD_VERSION for default analytic forces)")
