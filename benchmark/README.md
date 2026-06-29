# Benchmarks

Performance benchmarks for ACEpotentials, with a focus on force evaluation.

The benchmarks run in their own environment (`benchmark/Project.toml`, which
`develop`s the package at `..`), keeping perf-only dependencies out of the main
test path. They require the ACE registry:

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.activate("benchmark"); Pkg.instantiate()
```

## Files

| File | Purpose |
|------|---------|
| `common.jl` | Shared model/system setup + `measure` helper. Single source of truth. |
| `bench_forces_regression.jl` | Reproduce & quantify the autograd force regression: classic (analytic) vs ET, energy & forces, with allocations + a `site_grads` isolation timing. |
| `bench_crossversion.jl` | Compare a pinned **previous release** of ACEpotentials against the current checkout (release-vs-release drift). |
| `benchmarks.jl` | PkgBenchmark `SUITE` used by CI to guard against regressions. |
| `benchmark_full_model.jl` | Pre-existing full-model ACE vs ETACE comparison (CPU/GPU). |

## Reproduce the force regression locally

```bash
JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/bench_forces_regression.jl
```

The hypothesis (autograd `site_grads` dominates force cost) is confirmed when the
**force** ET/classic ratio is far larger than the **energy** ratio and ET force
allocations scale much worse than ET energy allocations.

## Cross-version comparison

```bash
julia --project=. benchmark/bench_crossversion.jl
```

This installs a previous release into a temporary, isolated environment and
benchmarks `forces` on the same systems, then compares to the current dev
checkout. Note the previous release's default `ace1_model` path was already
analytic, so this measures total release drift; `bench_forces_regression.jl`
isolates the autograd contribution within the current codebase.

## Regression testing in CI

`.github/workflows/Benchmark.yml` runs the PkgBenchmark suite via
[BenchmarkCI.jl](https://github.com/tkf/BenchmarkCI.jl), judging the PR head
against its merge base **on the same runner** (which cancels most hardware
noise). It is **non-blocking** (`continue-on-error: true`) until the threshold
is calibrated on real PRs; promote it to a required check once stable.

Run the suite locally:

```julia
using PkgBenchmark
r = benchmarkpkg("ACEpotentials"; script="benchmark/benchmarks.jl")
export_markdown(stdout, r)
```

Sanity-check the detector by temporarily restoring the Zygote-based `site_grads`
in `src/et_models/et_ace.jl` / `et_pair.jl`; the force benchmarks should regress
sharply.
