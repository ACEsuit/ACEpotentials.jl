# Force-evaluation performance regression — investigation & fix

## TL;DR

The v0.10 (EquivariantTensors) series had a **~13× force-evaluation slowdown** vs
the previous release (0.9.1) for the **default `ace1_model` / classic `ACEModel`**.

- The original hypothesis — that the slowdown came from the ET backend computing
  forces via **autograd** — is **refuted**. The ET autograd path is actually
  *faster* than the classic analytic path was.
- The real cause is a **type instability** in the classic analytic `evaluate_ed`
  (`src/models/ace.jl`): `EquivariantTensors.pullback` returns `Tuple{Any,Any}`,
  so the inline gradient-assembly loops ran with per-element dynamic dispatch.
- **Fix:** a one-function **function barrier** (`_assemble_grad_ed!`). Restores
  performance to ~0.9.1 levels; forces are bit-identical.

## How it was found

Benchmarks (`benchmark/`), single-threaded, parameter-matched models:

1. **In-repo, classic vs ET** (`bench_forces_regression.jl`): ET autograd forces
   were ~4× *faster* than classic analytic forces — opposite to the hypothesis.
   Classic forces were ~12.7× their own energy cost (ET only ~2.1×).
2. **Cross-version** (`bench_crossversion.jl`, 0.9.1 vs dev): classic forces were
   **11–14× slower** in dev than in 0.9.1.
3. **Profiling** `evaluate_ed`: numerical kernels (radial, Ylm, tensor fwd/bwd,
   pair) summed to ~30 µs, but the full call took ~400 µs and allocated 764 KB.
   Per-block timing pinned ~225 µs (80%) on the gradient-**assembly loops**.
4. **Isolation:** the same loops with concrete-typed inputs ran in **1.5 µs / 0 B**.
   `Base.return_types(EquivariantTensors.pullback, …) == Tuple{Any,Any}` →
   `∂Rnl`/`∂Ylm` reach the loop as `Any` → dynamic dispatch per element.

## The fix

`evaluate_ed` built `∇Ei` with the products `∂Rnl[j,t]*dRnl[j,t]*∇rs[j]` and
`∂Ylm[j,t]*dYlm[j,t]` **inline**. Because `∂Rnl`/`∂Ylm` are `Any`, every product
was dynamically dispatched. Moving the loops into `_assemble_grad_ed!` lets Julia
re-specialise on the concrete runtime types (a standard *function barrier*).

```
evaluate_ed, 40 neighbours:  386 µs  →  30 µs   (ΔE = 0, max|Δg| = 0)
```

## Results (single-thread)

Classic `forces`, before vs after the fix (Si/O, order 2):

| atoms | before | after | speedup |
|------:|-------:|------:|--------:|
|    64 |   22.0 ms |  2.69 ms | 8.2× |
|   512 |  190.8 ms | 18.3 ms | 10.4× |
|   800 |  298.3 ms | 29.2 ms | 10.2× |

Cross-version (Si, `ace1_model`), dev/0.9.1 ratio: **~13× → ~1.15×**.

All ACE model + calculator tests pass (finite-difference force checks included).

## Notes / follow-ups

- The residual ~15% vs 0.9.1 is minor (ET `pullback`/`evaluate` overhead vs the
  old EquivariantModels backend) — not pursued here.
- The same `Tuple{Any,Any}` return from `EquivariantTensors.pullback` could be
  fixed upstream in EquivariantTensors.jl; the function barrier is the robust
  in-repo workaround. `grad_params` (fitting path) already passes the pullback
  result through a function (`pb_Rnl`), so it is largely insulated.
- Regression guard: `benchmark/benchmarks.jl` (PkgBenchmark) run by the
  non-blocking `.github/workflows/Benchmark.yml` job (results posted to the job
  summary). Base-branch comparison (`judge`) can be enabled once the suite is on
  `main`; see `benchmark/README.md`.
