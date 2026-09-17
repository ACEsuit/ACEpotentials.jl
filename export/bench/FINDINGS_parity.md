# Finding: making `lammps-export` correct and fast — the `export/perf-parity` branch (2026-09-17)

**Where this file lives, and why here.** The plan asked for
`docs/findings/FINDINGS_lammps_export_parity.md` in the ACEpotentials `pr/lammps-throughput`
checkout *if that checkout is available on this host*. It is not: `~/si-ace/ACEpotentials` is
an rsync'd copy with no git. This is the plan's own stated fallback location.

**What it updates.** Two findings of 2026-09-15 — one on correctness, one on where the time
goes — written against `acesuit/lammps-export` at 075d3859. **They are quoted inline below
rather than linked**: they live in the controller's working directory, which is not part of
this repository, so a link would be dead for every reader of this branch. Every defect they
named is closed here, or is listed in §5 as open, by name.

> **From the correctness finding (2026-09-15), the four conclusions this document answers:**
>
> 2. *"The `:polynomial` export is exact."* Energies and forces of the E0 + many-body part
>    agree with the ETACE calculator and with the fitted classic `ACEModel` to
>    **max |dF| = 6.7e-14 eV/Å, |dE|/atom = 1.5e-13 eV** on forces of 3–6.7 eV/Å.
> 3. *"The `:hermite_spline` export — the README's recommended mode — is broken for any
>    multi-element model at the branch tip"*: one spline table per ordered species pair,
>    dispatched on a *symmetric* pair index, so for NZ ≥ 2 most neighbour species read the
>    wrong table. As-is: **13.5 eV/Å, 7.2 eV/atom** errors in LAMMPS. *"The branch's
>    multi-species test only exercises `:polynomial`, so CI cannot see this."*
> 4. *"Even fixed, `:hermite_spline` is not exact against the fitted model"*, because
>    `ETModels.splinify` is the inexact step: **2.7e-4 eV/Å at the default `Nspl = 50`,
>    3.0e-6 at `Nspl = 200`**. *"So the README's 'machine precision' claim is true only
>    relative to the splined model."*
> 5. *"The pair potential is not carried."* `export_ace_model(::StackedCalculator)` picks out
>    `ETOneBody` and `ETACE` and silently ignores an `ETPairModel`. *"The export of the
>    (E0, pair, ACE) stack is byte-identical to that of the (E0, ACE) stack… the E0 +
>    many-body part alone is attractive on every dimer (−1.5 eV at 3.0 Å, Cr–Cr) and changes
>    held-out forces by up to 6.9 eV/Å."*
>
> **From the performance finding (2026-09-15):**
>
> *"Measured on the five-element Cantor model (moriarty, one pinned core, loadavg 5–6):
> `pair_style ace` = 522 µs/atom (`:polynomial`), 195 µs/atom (`:hermite_spline`, dispatch
> fixed); `pair_style pace` (fork, same 1348-function basis, 1e4-node yace) = 75 µs/atom. So
> the exact mode is 7.0x and the recommended mode 2.6x behind ML-PACE on this host. In
> `:polynomial` mode 84 % of the site time is the dense radial mixing… None of this is
> architectural: every item is a code-generator change (days each, ~2 weeks in total), none
> needs EquivariantTensors or the C++ plugin to change… After all of it the per-site cost
> should land at roughly 30–40 µs on this host against ML-PACE's 75 — i.e. parity is
> plausible."*
>
> And, on multi-rank behaviour — the claim §4 sets out to test:
>
> *"In `benchmark/results/` the ETACE runs show `%varavg` on `Pair` of 34 % (Hermite,
> 2 ranks), 22 % (4), 8 % (8), 24.5 % (poly, 4 ranks), 2.4 % (poly, 8), while on the same
> 2000-atom B2 box the `pace` runs show 1.1–1.4 % at 2, 4 and 8 ranks… The decomposition is
> therefore balanced; the ETACE numbers are erratic in rank count and consistent with a
> shared, loaded host… One thing worth ruling out when it is re-measured: the embedded Julia
> runtime in each rank starts its own GC threads, so eight ranks on eight cores may
> oversubscribe (`ps -T` on a running rank would show it)."*

---

## Summary

The first finding concluded that at `acesuit/lammps-export` (075d3859) the exact
`:polynomial` mode reproduced only the **many-body** part of a physical five-element model —
the pair potential was silently dropped — that the recommended `:hermite_spline` mode was
**wrong for every multi-element model**, and that the route was **7x slower** than
`pair_style pace recursive`. The second finding predicted that ~30–40 µs/site was reachable by
code-generator work alone, in about two weeks, with no change to EquivariantTensors and none
to the C++ plugin.

Both predictions held.

- **The pair term is exported.** `max|dF|` against the full `(E0, pair, ETACE)` stack went
  from **6.878706 eV/Å to 2.966e-14 eV/Å**.
- **`:hermite_spline` dispatches correctly for `NZ ≥ 2`**, is labelled approximate everywhere
  it is documented, and now **refuses** the one configuration (per-pair cutoffs) whose
  reference model cannot even be evaluated.
- **The plan's acceptance gate — ≤ 1.2x `pace recursive` per core, both reference models,
  exact `:polynomial` — is met with margin, a task early.** Cantor and TiAl figures are in §1.
  Cantor `:polynomial` went from **512.2 to 58.1 µs/site**, past the plan's own 150 µs/site
  target; TiAl `:polynomial` from 424.6 to 92.8.
- **The blindness is gone too**, and that is the half of this work that will age best. The
  export suite went from 43 tests, several of which asserted nothing, to **32 549 passing across
  ten groups, with every group required to have actually run** (380 of them outside the DAG
  group's exhaustive sweep); four pieces of "looked green, asserted nothing"
  coverage were found and fixed *inside the tasks whose job was to remove that class*.

One thing did not pay, and ships off:

- **B3, the DAG re-association of the AA products, is implemented, exact and off by default.**
  It does what it was budgeted to do — the tensor step is 1.6x (Cantor) / 2.3x (TiAl) faster —
  and on Cantor it loses more than that back in the two neighbour passes it does not touch.
  `aa_products=:dag` is opt-in. **The negative result is the deliverable**, not a failure to
  hide: it is the clearest instance of the rule that an optimisation must show on both boxes.

Two questions are put to the maintainer rather than decided here: whether `:hermite_spline`
should be **retired** (§7), and what to do with the **inoperable minimal-export path** (§8).

---

## 1. What changed at each step, with the measured factor

All ratios are `pair_style ace` / `pair_style pace recursive`, same core, same session, on the
2048-atom Cantor and 2000-atom TiAl boxes, `timestep 0.0`, 100 steps, one MPI rank pinned with
`taskset`, `OMP_NUM_THREADS=1`. **Lower is better; below 1.00 is faster than ML-PACE.** The
protocol is in [`README.md`](README.md); the rows are in
[`artefacts/rows_task8.txt`](artefacts/rows_task8.txt) and every figure below is what
`summarise_rows.py --series both` prints from them.

| step | Cantor µs/site | vs `pace` | TiAl µs/site | vs `pace` |
|---|---|---|---|---|
| baseline (Task 4 generator) | 511.8 | **7.918x** | 429.6 | **2.300x** |
| B1 — radial mixing from `W`'s sparsity (Task 5) | 224.9 | **3.488x** | 281.7 | **1.504x** |
| **B2 — per-neighbour kernel (Task 6) — SHIPPED** | **58.1** | **0.893x** | **94.0** | **0.505x** |
| `:hermite_spline`, `Nspl = 50`, at B2 | 62.5 | 0.967x | 152.5 | 0.816x |
| `pair_style pace recursive` (comparator, pooled over every block on that box) | 64.6 | 1.00 | 186.8 | 1.00 |

**The gate is ≤ 1.20x on both models in exact `:polynomial`. Measured: 0.893x and 0.505x** —
and all four shipped configurations are faster than `pair_style pace recursive` outright.

Each ratio is that tag's own pooled ACE median over its own pooled comparator median, so
numerator and denominator come from the same blocks on the same core minutes apart; the
comparator row is pooled over every block on that box and is informational.

Taken 2026-09-17 on `moriarty`, core 31, `taskset`, `OMP_NUM_THREADS=1`, `timestep 0.0`, 100
steps per run; 29 blocks, 68 runs; `/proc/loadavg` 0.27–1.82 over the blocks that count (this
benchmark's own single pinned process is ~1.0). Every library passed its source, library,
LAMMPS and 2-rank gates before it was timed, enforced by a content-based interlock
(`gate=OK[...]` in every row). One block is excluded by name, with its reason in the rows file
— see below.

Every figure above is what

```
export/bench/summarise_rows.py export/bench/artefacts/rows_task8.txt --series both
```

prints, numerator and denominator alike. **Pass no tag arguments**, so the tool groups by
exact tag: passing `cantor_poly` as a prefix also matches `cantor_poly_b1` and `_b2` and pools
three different binaries into one statistic. That is the one trap in the interface.

These reproduce the per-task rows they supersede — 8.15 → 7.92, 3.51 → 3.49, 0.90 → 0.89,
0.97 → 0.97, 2.27 → 2.30, 1.54 → 1.50, 0.49 → 0.51, 0.82 → 0.82 — on a different day with the
whole chain re-measured in one session.

### Step by step

**Correctness first (Tasks 1–3), then speed.** No optimisation was taken until the thing being
optimised was the right computation, and every performance step keeps the previous generator's
exported model as its reference at **1e-13 relative**.

| step | what it did | Cantor `:polynomial` | TiAl `:polynomial` |
|---|---|---|---|
| Task 1 | export the `ETPairModel` term (it had been silently dropped) | correctness; costs **+6.5 %** | — |
| Task 2 | one *ordered* pair index everywhere; Hermite correct for `NZ ≥ 2`; both `agnesi_transform` bugs | correctness | correctness |
| Task 3 | LAMMPS/MPI gates on a pair-bearing model; CI that fails instead of skipping | coverage | coverage |
| Task 4 | the TiAl order-4 second reference model; the gated benchmark harness | **8.15x** baseline | **2.27x** baseline |
| **Task 5 (B1)** | radial mixing from `W`'s structure; prune unused `(n,l)` rows and per-pair zero rows; integer transform powers | 8.15 → **3.51x** (2.28x faster) | 2.27 → **1.54x** (1.52x faster) |
| **Task 6 (B2)** | per-neighbour stack kernel; species-block `A` accumulation; forces from `∂A`; workspace API; neighbour cap removed | 3.51 → **0.90x** (3.88x faster) | 1.54 → **0.49x** (3.09x faster) |
| Task 7 (B3) | DAG `AA` products with a `C̃`-seeded backward | **off by default** — 1.19x, i.e. 1.32x *slower* than B2 | not gateable, not shipped |

**B1 is a sparsity result.** Both reference models mix the radial basis with a weight tensor
`W[n, q, pair]` that is 99.5 % zeros (`init_Wradial = :onehot`); the generator had been
emitting two dense `74×45` static matrix–vector products per edge and streaming 26 KB out of a
666 KB weight table for every one of ~411 k edges per step. Emitting only the rows that are
both nonzero and read removed 94 % (Cantor) / 88 % (TiAl) of the `:polynomial`-vs-Hermite gap
and nothing else — exactly what it targets, and **both Hermite rows moved by 0**, because a
Hermite edge reads four knots and never had that traffic to lose.

**B2 is two changes that only pay together**, and the task report says so rather than inventing
a split. The recurrence is truncated to the polynomials any `W` column actually reads —
**45 → 6 on Cantor, 33 → 11 on TiAl**, exact because the recurrence is strictly forward — and
the per-neighbour work moves from `[MAX_NEIGHBORS, feature]` arrays walked with a 2 KB stride
into a per-edge stack kernel. B1 already had the narrow tables and got nothing from them,
because the narrow result was scattered back to full width immediately; the width only pays
once it travels with the kernel.

**B3 is the negative result.** Per-phase profile — an attribution tool, not a protocol row:

| phase | Cantor `:flat` | Cantor `:dag` | TiAl `:flat` | TiAl `:dag` |
|---|---|---|---|---|
| embed (pass 1) | 20.82 µs | 31.39 (**1.51x**) | 10.64 | 10.27 (0.97x) |
| **tensor step** | 13.49 | **10.98** (0.81x) | 61.92 | **25.18** (**0.41x**) |
| forces (pass 2) | 28.50 | 42.67 (**1.50x**) | 16.70 | 15.54 (0.93x) |
| whole site | 49.60 | 65.18 (1.31x) | 83.57 | **46.00** (0.55x) |

The DAG hits its target on both models, and on Cantor the two **byte-identical** neighbour
passes get 27–29 % slower around it: a 201-neighbour site runs them either side of the DAG's
48 kB of gathered and scattered working set, and the per-edge tables are evicted. TiAl's
112-neighbour site, whose tensor step is 73 % of the total, has nothing to lose and the site is
1.71x faster. It is a cache-footprint loss on the dense-neighbour model, not a failure of the
idea — and a model shaped like TiAl would benefit, which is why the code ships (opt-in) rather
than being deleted.

---

## 2. Exactness at every level

Measured on the shipped generator, `aa_products=:flat`, read from the gate manifests
(`bench_parity/<tag>.gated`, written by `verify_bench_models.jl` and `gate_bench_libs.jl`).
Forces are absolute eV/Å; energies are per atom unless marked relative.

| level | reference | tol | Cantor `:polynomial` | TiAl `:polynomial` |
|---|---|---|---|---|
| generated Julia | the fitted `(E0, pair, ETACE)` stack | 1e-12 | dE 1.658e-14, **dF 1.840e-14**, dV 1.557e-13 | dE 2.274e-13, **dF 5.799e-13**, dV 6.928e-13 |
| compiled `.so` (Python C API) | the generated Julia | 1e-12 | dE 1.819e-15, dF 4.232e-14 | dE 6.985e-13, dF 5.946e-13 |
| `pair_style ace` in LAMMPS | the compiled library | 1e-10 | dE 1.273e-14, dF 1.539e-14 | dE 1.513e-12, dF 5.818e-13 |
| 2 MPI ranks vs 1 | the serial run | 1e-13 rel / 1e-12 abs | dE **0.0**, dF 3.210e-15 | dE **0.0**, dF 5.595e-14 |
| `OMP_NUM_THREADS=4` vs serial | the serial run | bitwise | dE **exactly 0**, dF 2.96e-15 | dE **exactly 0**, dF 8.59e-14 |
| generator vs previous generator | the previous commit's export | 1e-13 rel | **bit-identical** | **bit-identical** |

And for `:hermite_spline` (`Nspl = 50`), against the **splinified** model, which is the only
reference it can be held to:

| | Cantor `:hermite` | TiAl `:hermite` |
|---|---|---|
| generated Julia vs splinified | dE 1.895e-14, dF 1.931e-14 | dE 2.695e-13, dF 5.357e-13 |
| compiled `.so` vs Julia | dE 9.095e-15, dF 3.902e-14 | dE 0.0, dF 6.011e-13 |
| LAMMPS vs library | dE 5.457e-15, dF 1.690e-14 | dE 1.746e-12, dF 5.695e-13 |
| **vs the FITTED model — reported, never asserted** | **2.679e-4 eV/Å** (2.952e-6 at `Nspl = 200`) | **1.646e-2 eV/Å** |

That last row is the whole story of `:hermite_spline`: it is exact against the model it *can*
be exact against, and that model is not the one that was fitted. The suite prints the number on
every run and asserts nothing about it, which is the honest arrangement — it is **model** error
introduced by `splinify()`, not export error.

### Two tolerances that sit below double-precision resolution, and were not moved

Four times in this plan an **absolute** tolerance turned out to be the wrong instrument for a
quantity whose magnitude varies by orders of magnitude between models. Each was resolved by
changing the *metric* or by *documenting* — never by loosening a number:

1. **Virial, absolute (Task 0).** The virial is extensive (~1.8e3 eV over ~4300 edges), so an
   absolute 1e-12 gate is 4 ulp of double precision. It is compared **per atom**, like the
   energy. Measured deviation 5.9e-12 absolute = 8e-16…3.6e-15 *relative*.
2. **Rank-to-rank energy, absolute (Task 4).** `E0(Ti) = -1586` against `E0(Cr) ≈ -14` makes
   the same per-atom gate two orders tighter on TiAl. Changed to **relative** (1e-13 of `|E|`),
   forces still absolute. `sum(E0)` is explicitly **not** subtracted — removing a term to make
   a comparison pass is the weakening the constraints forbid, whereas changing the metric's
   dimension is the honest fix. Verified from the code by an independent reviewer, and the
   threshold is a real gate: any genuine decomposition fault (a missing ghost atom, a
   half/full list mix-up, a double-counted pair) perturbs the total by at least one pair
   interaction, 500x to 5e7x above it.
3. **TiAl virial parity (Task 6).** κ = 428–970 puts the achievable floor at κ·ε ≈ 2.15e-13,
   *above* the 1e-13 gate. One κ-aware rule — `3e-13` for both TiAl modes — replaced an
   exemption keyed by *case name* that would otherwise have applied silently forever.
4. **TiAl force, absolute, 1e-12 (Task 7) — documented, not changed.** At 256-bit precision,
   on TiAl Ti: κ = 18.8, floor 1.154e-12 (Al: κ = 37.7, floor 1.228e-12), and **the shipped
   generator's own error against exact arithmetic is 1.307e-12 — larger than the 1e-12 gate it
   passes.** It passes because it shares EquivariantTensors' product association, so on that
   model the gate measures *agreement with the reference's association* rather than accuracy.
   On Cantor the `:dag` route is the *more* accurate of the two on all five species. **The gate
   was left at 1e-12**: the shipped default passes it, and the only thing it rejects is a route
   we chose not to ship for independent reasons, so moving it to admit something we are not
   shipping is pure downside. There is a comment at all three sites that apply it, opening
   "READ THIS BEFORE CONCLUDING THAT A CHANGE WHICH JUST MISSES THIS GATE IS WRONG".
   **A change that just misses 1e-12 on the TiAl order-4 model is not automatically wrong.**

**No tolerance was loosened anywhere in this work.** Several were tightened: every Hermite `≈`
gate lacked `rtol=0`, so `isapprox`'s default *relative* tolerance had made `atol=1e-12`
decorative.

---

## 3. The must-fix items from the first finding, closed

| # | item | status |
|---|---|---|
| 1 | **The pair term is not carried.** `export_ace_model(::StackedCalculator)` silently ignored `ETPairModel`; the (E0, pair, ACE) export was byte-identical to the (E0, ACE) one. On this model the pair term carries the repulsive core. | **Closed (Task 1).** Emitted in both radial modes. `max|dF|` 6.878706 → 2.966e-14 eV/Å. The exporter now also *refuses* a calculator it does not understand instead of quietly dropping part of it. |
| 2 | **Hermite dispatch is wrong for `NZ ≥ 2`.** One table per *ordered* pair, dispatched on a *symmetric* index: 13.5 eV/Å, 7.2 eV/atom in LAMMPS. | **Closed (Task 2).** One ordered `pair_idx` keys every per-pair table; `zz2pair_sym` is gone. Proved non-decorative by mutation: substituting the symmetric index moves `max|dF|` to 66.2 eV/Å (`:polynomial`) / 56.6 (Hermite). |
| 3 | **`MAX_NEIGHBORS = 256`**, a silent cap. | **Closed (Task 6).** Gone. A workspace is sized from the *model* (`N_A`, `N_AA`, `N_BASIS`) and never from the neighbour count, so a site of any size uses the same buffers. |
| 4 | **Re-entrancy — the OpenMP path was unsafe.** Mutable global scratch in the library. | **Closed (Task 6).** No mutable global state. Evaluation takes an opaque, *tagged*, validated workspace handle from an image-resident pool of 32; one per OpenMP thread in the plugin, one per `ACELibrary` instance. `OMP_NUM_THREADS=4` vs serial is **bitwise** on both models. Validation covers tag, range **and** `WORKSPACE_TAKEN`, so use-after-free and use-of-never-allocated are rejected rather than served. |
| 5 | **CI blindness.** The multi-species test exercised only `:polynomial`, so CI could not see defect 2. | **Closed (Task 3) — and it was worse than reported.** Four pieces of dead coverage: the *entire* MPI group had been skipping silently (`which mpirun` finds nothing on this host); `which lmp` resolved to a pip LAMMPS that segfaults before the plugin is reached, so the repo's own plugin was never tested; a 1e-6 Python comparison was guarded on a key nothing ever set; and `test_hermite_accuracy.jl` was run by nothing at all. Skips now route through `skip_group`, so a skip is a visible `Broken` and a hard failure under `ACE_REQUIRE_GROUPS`. |
| 6 | **Tolerances.** Tests asserted forces to 1e-8; "machine precision" was claimed for an approximate mode. | **Closed (Tasks 2–3).** Gates are as tabulated in §2; every Hermite gate is `rtol=0`; the false "machine precision" and "3–4x faster" claims are deleted repo-wide; `:polynomial` is the documented default; and a splinified model exported *without* asking for Hermite is now a hard **error**, not a silent substitution — the reverse mismatch stays a warning, since substituting the exact mode for the approximate one cannot make a result wrong. |

**The one that was found rather than inherited.** Task 6's first B2 passed *every* gate —
source 1e-12, library, OpenMP, parity, `juliac` — and then crashed after about one step of a
real LAMMPS run: a runtime-allocated Julia object is not reliably rooted in a `juliac --trim`
image. The fix is the image-resident pool. The gate that was missing is now **L1**
(`liveness_gc.py`): it drives the library until `ace_gc_count()` reports three real
collections, requires every result to stay **bitwise** equal, and fails if the collections
never happen — so it cannot pass vacuously at any cell size. It would have caught that crash on
all four tags with margin. The sizing behind it was settled by building an instrument
(`ace_gc_count`/`ace_alloc_bytes` exported from the library) after two independent estimates
disagreed in opposite directions: a site call costs ~`56n + 208` bytes, and the first
collection lands at 43.1 MiB — Julia's default collect interval.

---

## 4. Multi-rank sanity

4 ranks, the 2000-atom TiAl box, 100 steps, `pair_style ace` against `pair_style pace
recursive` on the identical box and decomposition. Artefact:
[`artefacts/mpi4_task8.txt`](artefacts/mpi4_task8.txt); script `export/bench/mpi_sanity.sh`.

| | `Pair` %varavg | `Pair` %total | `Comm` %varavg | loop |
|---|---|---|---|---|
| `pair_style ace`, mpirun defaults | **30.9, 35.1** | 73–79 % | 51, 68 | 6.9–7.8 s |
| `pair_style ace`, `--bind-to core --map-by core` | **12.3, 22.1, 24.6** | 79–85 % | 37–49 | 5.8–7.0 s |
| `pace recursive`, same box | **4.0, 5.2, 5.8, 6.3, 6.8** | 98 % | 29–42 | 10.6 s |

**The plan expected "a few percent, like `pace`'s". It is not a few percent, and the earlier
finding's attribution of 22–34 % to host load does not survive being tested** — this host was
quiet and the figure reproduces across runs and rank bindings.

**It is not the decomposition.** Both pair styles report the same work split, balanced to
1.1 %: `Nlocal` 496–507 (ave 500) and `FullNghs` 55 618–56 835 (ave 56 065). The domain
decomposition is even; what varies is how long a rank spends inside `Pair`.

**What it is, as far as the evidence goes.** Two things, one demonstrated and one strongly
indicated:

1. **The ranks are not single-threaded, even at `OMP_NUM_THREADS=1`.** `ps -T` on a live
   4-rank run shows **1, 3, 5 and 5 threads** across the four ranks — the embedded Julia
   runtime's own — placed by mpirun's default policy on cores it has also handed to other
   ranks (two ranks' threads were observed on core 0 simultaneously). This is exactly the
   check the earlier finding said was worth doing and had not been done. Binding removes about
   a third of the effect (30.9 → 22.1 %) and takes ~10 % off the loop time.
2. **The residue looks like independent per-rank garbage collection.** Each rank's library
   allocates ~`56n + 208` bytes per site call and collects on its own schedule (first
   collection at 43.1 MiB — measured, §3), with nothing synchronising the four. A pause inside
   `Pair` on one rank leaves the others waiting in `Comm`, and that is precisely the shape of
   the data: the per-rank times are **complementary to the millisecond** — min `Pair` 4.989 +
   max `Comm` 2.832 = 7.82 s = max `Pair` 6.592 + min `Comm` 1.229. The `%varavg` figure is
   also not stable run to run (12.3 / 22.1 / 24.6 under identical conditions) and does not
   average out over 4x more steps, which random pauses explain and a systematic per-rank cost
   difference does not. Restricting the runtime's GC threads (`JULIA_NUM_GC_THREADS=1`) gave
   18.5 %, the lowest bound figure seen, but on a slower loop — suggestive, not conclusive.

**What it does and does not mean.** It is a *throughput* item, not a correctness one: the
2-rank gate is exact (energy 0.0 relative, forces 5.6e-14) and the 4-rank run computes the same
physics. Even carrying a 25–35 % imbalance, `pair_style ace` completes the 4-rank loop in 6.9 s
against `pace`'s 10.6 s. But it will get worse with rank count on a fixed box, and it is the
clearest remaining performance item after B2 — a bigger one than anything left in the kernel.
**Recommended next step:** run with explicit binding, and investigate whether the library can
be made to allocate nothing per site call (the workspace already exists; the ~`56n` is
per-call temporaries), which would remove the GC from the hot path entirely.

---

## 5. What remains open

1. **The Julia runtime ships beside LAMMPS.** ~20 MB of `libjulia` and support libraries on
   `LD_LIBRARY_PATH`. No Julia process starts and no Julia code is interpreted, but the runtime
   is not removable. Unchanged by this work and not addressable inside it.
2. **`:hermite_spline` is approximate by construction** (§2) — and, since B2, also *slower*
   than the exact mode on both reference models. See §7.
3. **`ace_site_basis` is kept and works.** Task 7 proposed dropping it; it was kept, because
   the ABI is consumed by `ase-ace` and the plugin and silently removing an exported symbol is
   a breaking change the spec did not authorise. The `A2BMAP_*`/`WB_*` constants it needs are
   emitted **only** when `for_library = true`, and the energy/force path no longer goes through
   `B` at all. Cost: a few KB of unused constants in libraries that never call it.
4. **The minimal-export path is inoperable.** See §8 — a maintainer decision.
5. **The CI changes were verified locally, never on a hosted runner.** Every CI fix in Task 3 —
   including the artifact gap that made the pair-bearing 1e-10 LAMMPS gate *error* rather than
   run — was checked on this host by moving files aside and watching the job fail with a named
   reason. Nothing has been pushed, so nothing has run on GitHub. **This is the largest
   unverified item on the branch.**
6. **Two unexplained timing observations**, both recorded rather than smoothed away, and both
   conservative in direction:
   - a single `cantor_poly` block that ran ~10.9 % above the six-run median that followed it,
     with only 1.28 % internal spread. The cold-start hypothesis was **tested over two sessions
     and did not reproduce** (first-block deltas −0.85 % and +1.76 %, opposite signs, against
     an all-run span of 2.8 %), so no warm-up discard was added to the protocol; the block is
     excluded by name in its own rows file with the reasoning written beside it.
   - the micro-profile's Cantor `:dag` column moved ~17 % when the script behind it was
     committed and re-run, while the `:flat` column reproduced within run-to-run spread. The
     correction *strengthened* the conclusion. **Quote protocol rows, not the micro-profile**:
     its ratios are stable, its absolutes are not.
7. **`benchmark/accuracy_test.jl:69` and `benchmark/julia_benchmark.jl:77`** call
   `export_ace_model` with `radial_basis=:spline` and `n_spline_samples` — neither the symbol
   nor the keyword exists, and both scripts fail immediately. Pre-existing rot; `benchmark/` is
   outside this work's `export/`-only scope, so it is reported, not fixed.
8. **Multi-rank load balance: `Pair` %varavg is 12–35 %, against `pace`'s 4–7 %** on the same
   box with the same decomposition (§4). The work is balanced to 1.1 %, so it is the pair
   style's *time*, not its share of atoms. Demonstrated contributor: the embedded Julia
   runtime's per-rank threads sharing cores. Indicated contributor: unsynchronised per-rank
   GC. **This is now the largest performance item left**, larger than anything remaining in
   the kernel, and it is a throughput cost only — the 2-rank correctness gate is exact.
9. **`_bad_handle()` prints one stderr line per bad call, per site, per step.** Under a
   mismatched-ABI plugin that is a per-atom flood. It is a loud failure rather than a silent one
   — Task 6 confirmed the mismatch surfaces as a clean LAMMPS stop — but it should rate-limit.

---

## 6. Deferred minors, triaged

Every minor deferred anywhere in this plan, with a verdict. **Blocking** items are fixed on this
branch; **ship** items are real but do not block a merge. None is dropped silently.

| # | item | verdict |
|---|---|---|
| 1 | `check_export`'s `tol` keyword defaults to 1e-12 where the brief made it mandatory | **ship.** Every call site passes it explicitly, and a default equal to the constraint cannot weaken a gate. |
| 2 | Task 0 diagnosed the Python/LAMMPS group failures unasked | **ship.** No files were committed for it, and it was correct input to Task 3. |
| 3 | `task-2-brief` says "add `check_export_report`" when Task 0 had already built it | **ship.** Plan-document inconsistency only; resolved at dispatch. |
| 4 | an in-code comment said the 1e-12 gate is "~20x" the measured divergence; it is ~23.5x | **fixed** — the text no longer makes the claim. |
| 5 | `export/ase-ace/README.md:255` "Unix Sockets (Faster for Local)" — an unmeasured comparative heading | **ship, but flagged.** Same genre as the "3–4x faster" claim Task 2 deleted repo-wide. Pre-existing, and socket-vs-JuliaCall was never in scope to measure. It should either be measured or lose the word. |
| 6 | `@test s.d_se == snp.d_se` — exact equality between two independently measured maxima | **ship.** Correct today; it will break on an unrelated SpheriCart or compiler change rather than on the regression it guards. Worth relaxing to a relative comparison when someone next touches that file. |
| 7 | `test_multispecies.jl:265` asserts on a literal error-message substring | **ship.** A reword breaks the test loudly, which is the acceptable direction. |
| 8 | `verify_bench_models.jl` wrote a literal `source_gate=PASS` | **fixed.** Now derived from the three measured maxima. `check_export` throws first, so the literal was in fact always true — but "printed rather than computed" is the shape of two real defects already found here, and deriving it is one line. |
| 9 | `model_sha256` is recorded in the gate manifest and read by nothing | **ship.** Library↔source provenance is carried by `EXPORT_BUILD_ID`/`ace_build_id`, which *is* checked (`runtests.jl` refuses the library groups on a mismatch); `model_sha256` is a human-readable duplicate. |
| 10 | a failed series emits empty statistic fields beside its `FAILED` marker | **ship.** The row says `FAILED`, `bench_parity.sh` exits nonzero, and `summarise_rows.py` does not match such a row at all. |
| 11 | `export/lammps/test/compare_dump.py` keeps the absolute per-atom energy convention that gate C rejected | **ship — already documented.** The file carries the derivation of why per-atom is right *for the boxes its callers use*, and what would change it. |
| 12 | every artefact backing the published rows is untracked | **fixed.** `export/bench/artefacts/` now holds the close-out rows, the session transcript, the byte-comparison log and the MPI log, with a README saying what produced each. The historical `rows_task4…7.txt` remain in the untracked `bench_parity/`; every number quoted from them is superseded by the committed close-out table. |
| 13 | `M2`'s assertion compares the *emitted* `RNL_USED` against the *emitted* `ABASIS_SPEC`, both derived from one source | **ship.** Recorded at the time as catching a codegen/round-trip divergence, not as proving pruning safety; the parity gate and the 1e-12 export gates do that. |
| 14 | the `:dag` dedent was not actually fixed, and its comment stated a **false rule** | **fixed** (first commit of this task). The rule is now the one that holds, verified by running it: a leading newline is no defence, because Julia strips the *common* indent; only a column-0 line inside the literal pins it at zero. |
| 15 | `1.306e-12` should read `1.307e-12` in the three gate comment blocks | **fixed.** It is the number the whole comment turns on. |
| 16 | "the `:flat` column reproduces exactly" overstates | **fixed.** It reproduces *within run-to-run spread* — three of four phases inside the tool's own two-run range — while the `:dag` whole-site figure is **3.3x outside** it. |
| 17 | the byte-comparison script proving byte-identity was never committed | **fixed.** `export/test/bytecmp_generator.jl`, with its log in `artefacts/`. |
| 18 | the README provenance note sits ~107 lines below the rows it describes | **fixed.** Moved to the rows. |
| 19 | Task 7's report quotes a third, unlogged profile run in its µs table | **ship.** The committed log holds two runs and the README quotes ranges across all three; the conclusion does not depend on which is quoted, and §5.6 tells readers not to quote the profile's absolutes at all. |
| 20 | CI changes never confirmed on a hosted runner | **ship — but see §5.5.** It cannot be closed without pushing, and nothing is pushed. |
| 21 | the minimal-export path is inoperable | **maintainer's call — §8.** |
| 22 | `_bad_handle()` has no rate limit | **ship — §5.8.** |

Three further deferrals from the ledger were closed inside the plan and are recorded here only
so the list is complete: the silent `:polynomial → :hermite_spline` promotion (made a hard
error in Task 3), the three latent skip-or-pass guards in `test_mpi.jl` (a deferral the
controller **reversed** after a reviewer showed two of them were reachable, and fixed), and the
stale `libace_test.so` build stamp (picked up by Task 6, where the hazard turned out to be live
in this tree and the new stamp caught it).

---

## 7. Question 1 for the maintainer: should `:hermite_spline` be retired?

**The case for retiring it is far stronger than it was, and exactly one structural argument
against survives. The decision is the maintainer's; here is the evidence.**

### It has lost every advantage it was kept for

- **Speed — it is now the *slower* mode on both reference models.** Its stated justification
  was never speed in the abstract but "learned radials with a small `N_POLYS`", where a knot
  table beats a recurrence. B1 and B2 removed that: the recurrence is now emitted only at the
  width the model reads — **45 → 6 polynomials on Cantor, 33 → 11 on TiAl** — so the work
  Hermite existed to avoid is largely no longer done. See the table in §1. The old
  "3–4x faster" claim is not merely unmeasured, as Task 2 found when it deleted it; it is
  **inverted**.
- **Accuracy — it is approximate by construction, on the model people actually have.**
  2.679e-4 eV/Å on Cantor at `Nspl = 50` and **1.646e-2 eV/Å on TiAl**, against an exact mode
  gated at 1e-12 (§2).
- **Coverage — it is refused outright for per-pair cutoffs.** Not a policy choice: the
  *reference* model throws a `BoundsError` in upstream `EquivariantTensors._spl_grid` when an
  edge beyond a pair's own cutoff maps to `y = 1` exactly, so such an export is unverifiable by
  construction. The exported code itself clamps and is fine — but there is nothing to check it
  against. `EquivariantTensors` may not be modified within this work's constraints, so the
  export is refused with an error naming the offending pairs, and the test suite locks the
  upstream crash with `@test_throws BoundsError` so that it fails the day upstream is fixed.

### The "learned radials" justification is now provably empty — established, not assumed

The plan required this to be *shown*, so it was. Task 5 added a dense `RBASIS_W_k` branch, and
the parity suite carries a `dense_model()` case built with `init_Wradial = :glorot_normal` and
`Winit = :glorot_normal` — an **arbitrary dense** weight tensor, fully populated and asymmetric
per ordered pair. (Random, not fitted: what it exercises is the dense code path, and the
structural argument below is what carries the claim to genuinely learned weights.) That case,
`dense_poly`, exports through **`:polynomial`**, is gated at 1e-12 against its own stack, and
is **bit-identical** across generator revisions.

The structure guarantees it in general, not just for that fixture. The ETACE radial embedding
that `ace_learnable_Rnlrzz` builds is `EmbedDP(agnesi transform, orthogonal-polynomial
recurrence, SelectLinL W)`, i.e.

    Rnl[n] = env(r) · Σ_q W[n, q, pair] · P_q(y(r))

for **every** such model. "Learned" means `W` is learned; `P_q` is the same recurrence in every
case, and the generator reads `A`, `B`, `C` straight out of the basis's own `refstate`. Since
`:polynomial` now emits an arbitrary dense `W` exactly, **`:polynomial` covers every learned
radial — exactly, rather than approximately.**

### The one argument that survives, and it is the real question

`:polynomial` **cannot** export a model that has already been splinified. That is not a gap in
the generator: `ETModels.splinify` *replaces* the radial embedding with `ET.TransSelSplines`,
and there is no polynomial recurrence left to emit. The exporter refuses such a model rather
than substituting silently, which is correct. So:

> **`:hermite_spline` is the only export path for a splinified model. Retiring it means
> dropping support for deploying splinified models.**

And there is one workflow where that matters and the approximation vanishes entirely: **fitting
*after* splinification**, the recipe written out in the header of `export/src/splinify.jl`. Fit
on the splines and the splinified model *is* the fitted model — the Hermite export then
reproduces its own reference at 1e-12 and there is no model error at all. (No runnable example
in this repository follows that order: `examples/etace_lammps_tutorial.jl` and
`verify_cantor/chain_cantor.jl` both fit first and splinify afterwards, so the models they
produce do carry the errors in §2. The tutorial now says so at its Step 6.)

### The question, stated precisely

Splinification as a *deployment* optimisation has no rationale left — the exact mode is now
faster than the approximate one. So the question is narrower than "retire or keep":

> **Does ACEpotentials intend to keep supporting the fit-on-splines workflow as a deployable
> path?**
>
> - **If yes:** keep `:hermite_spline` and re-document *why*. Its use case is not "learned
>   radials with a small `N_POLYS`" (now false) and it is not speed (now inverted). It is
>   *"the model was fitted after `splinify()`, so the spline model is the reference"*. The
>   per-pair-cutoff refusal stays as it is, and the upstream `_spl_grid` bug should be raised.
> - **If no:** retire `:hermite_spline` and `splinify`-for-export together, in one deprecation.
>   Retiring the mode alone would leave `splinify()` producing models that nothing can export.

**Nothing here retires it.** Removing a documented, exported mode is a product decision, and
the spec explicitly chose to keep it fixed and non-default — so it is kept. What *was* required
and is done: the README, the mode table and the docstring state plainly, with the measured
numbers, that as of B2 Hermite is **both slower and approximate** on both reference models, so
that nobody selects it for speed.

---

## 8. Question 2 for the maintainer: the minimal-export path

**It does not work in this tree, and it did not work before this plan started.**

`pair_ace_minimal.cpp` loads a Julia file named by `ACE_C_INTERFACE_PATH`, and the file it
wants is `ace_c_interface_minimal.jl`. That file existed (added in `728107ac` / `1e88c697`) and
was **deleted in `0372f90d`** — before any of this work. A *different* file,
`export/src/ace_c_interface.jl`, survived until Task 6 deleted it: 312 lines, `include`d by
nothing, carrying a copy of the `ace_site_*` entry points **without the workspace handle** —
precisely the stale ABI that the tagged handles now exist to reject. Leaving it in place was a
standing invitation to copy the wrong signatures into new code.

Two coherent outcomes, and it is the maintainer's choice which:

1. **Restore it against the current ABI.** Every evaluation entry point takes `void *ws` first;
   `ace_workspace_new` / `ace_workspace_free` / `ace_max_workspaces` must exist; handles are
   tagged and validated. Note that the multi-model `model_id` design the old documentation
   describes is **not** compatible with one-model-per-compiled-library, so this is a rewrite,
   not a file restoration.
2. **Remove the whole minimal path** — `export/lammps/plugin/src/pair_ace_minimal.{cpp,h}`,
   `aceplugin_minimal.cpp`, `CMakeLists_minimal.txt`, `build_minimal.sh`, and the
   "C Interface API for Minimal Export" half of
   [`../docs/C_INTERFACE_API.md`](../docs/C_INTERFACE_API.md). It is a separate build that the
   main plugin's CMake does not reference, so removing it touches nothing that works.

Until that is decided, the documentation says at the top of that file that there are two ABIs
and that the second one's implementation is gone. **Nothing in this branch depends on it.**

---

## 9. How to reproduce anything here

```bash
# the full test suite, with a missing fixture a FAILURE rather than a skip
cd export/test && ACE_REQUIRE_GROUPS=all julia --project=.. runtests.jl

# the generator-vs-generator parity gate (EXPORT_REF_SHA is mandatory; there is no default,
# deliberately -- a gate that picks its own reference is worse than one that refuses to run)
cd export/test && EXPORT_REF_SHA=<commit> julia --project=.. runtests.jl parity

# the shipped default's generated SOURCE vs a reference commit, byte for byte
EXPORT_REF_SHA=b826c831 julia --project=export export/test/bytecmp_generator.jl

# the close-out measurement table (one session on an idle core)
export/bench/run_task8_table.sh

# quote a row -- never read a number off a line by eye
export/bench/summarise_rows.py export/bench/artefacts/rows_task8.txt --series both

# multi-rank load balance
export/bench/mpi_sanity.sh tial_poly_b2 4 100
```

Artefacts for every number in this file are in [`artefacts/`](artefacts/); the protocol, the
standing measurement rules and the historical per-task rows are in [`README.md`](README.md).
