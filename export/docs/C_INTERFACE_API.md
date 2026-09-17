# C Interface API

This file documents ONE C interface: the ABI of a compiled ACE model library.

| | exported/compiled library |
|---|---|
| produced by | `export_ace_model(...; for_library = true)` + `juliac --trim=safe` |
| one model per | shared library (`libace_<model>.so`) |
| consumed by | the LAMMPS `pair_style ace` plugin, `ase_ace.ACELibraryCalculator` |

**There used to be a second, incompatible ABI here — the "minimal export" — and it has been
REMOVED.** It is recorded rather than silently dropped, because a reader who arrives from an
older commit, an older branch or a stale bookmark needs to know which of the two they are
looking at.

It never worked in this tree. `pair_ace_minimal.cpp` loaded a Julia file named by
`ACE_C_INTERFACE_PATH`, and the file it wanted, `ace_c_interface_minimal.jl`, was deleted in
`0372f90d` — before any of the parity work began. A *different* file,
`export/src/ace_c_interface.jl`, survived until Task 6 deleted it: 312 lines, `include`d by
nothing, carrying a copy of the `ace_site_*` entry points **without the workspace handle** —
precisely the stale ABI the tagged handles in the compiled library exist to reject.

The maintainer's answer to `export/bench/FINDINGS_parity.md` §8 was to remove the whole path.
Gone with it: `export/lammps/plugin/src/pair_ace_minimal.{cpp,h}`, `aceplugin_minimal.cpp`,
`CMakeLists_minimal.txt`, `build_minimal.sh`, the two committed `.o` build products beside
them, `export/lammps/examples/in.ace_minimal_test`,
`export/lammps/test/test_lammps_minimal.jl`, `export/lammps/test/test_lammps_direct.jl`, and
the "C Interface API for Minimal Export" half of this file. It was a separate build that the
main plugin's CMake never referenced, so removing it touched nothing that works.

**If you want the multi-model, `model_id`-keyed design the removed half described, note that
it is not compatible with one-model-per-compiled-library.** Reviving it would be a rewrite
against the ABI below, not a file restoration.

---

# Compiled-library ABI (`@ccallable`)

The symbols `Base.@ccallable`-exported by a generated model file. All of them are plain C:
no Julia runtime call is needed beyond loading the library.

## Workspaces and re-entrancy

Since the per-neighbour kernel (B2) the library holds **no mutable global state**. All scratch
lives in a *workspace*, which the caller obtains and passes to every evaluation entry point as
its **first argument**:

```c
void  *ace_workspace_new(void);   /* NULL when the pool is exhausted */
void   ace_workspace_free(void *ws);
int    ace_max_workspaces(void);  /* size of the pool, fixed when the library was built */
```

* The library is **re-entrant given one workspace per concurrent caller**. It is **not**
  thread-safe with a shared workspace, and there is no internal locking that would make it so:
  two threads in one workspace corrupt each other's `A`, `AA` and `∂A`.
* **The pool is FIXED SIZE and is built into the library.** `ace_max_workspaces()` returns it
  (currently 32 for a model exported by this generator). `ace_workspace_new` returns **NULL**
  once that many are outstanding — check the return value, and size your thread pool with
  `ace_max_workspaces()`. It is not a bound on anything physical; to raise it, change
  `MAX_WORKSPACES` in `export/src/write_c_interface.jl` and re-export the model.
* A workspace is **sized from the MODEL** (`N_A`, `N_AA`, `N_BASIS`) and never from the
  neighbour count. It does not grow, and it is never reallocated. There is consequently **no
  maximum neighbour count**: the pre-B2 `MAX_NEIGHBORS = 256` cap is gone, and a site of any
  size uses the same buffers.
* `ace_workspace_new` / `ace_workspace_free` take an internal lock (they flip a slot's
  *taken* flag in the shared pool), so they may be called from anywhere, including inside a
  parallel region. The evaluation entries take **no** lock. Allocate every workspace *before*
  concurrent use all the same.
* Handles are **tagged opaque integers**, not pointers into the library's heap, and they are
  validated on every call: passing something that is not a handle is rejected rather than
  served. Freeing is not required before `dlclose()` (the pool is part of the image), but it
  is good practice and it returns the slot.

WHY THE POOL IS IN THE IMAGE, in case a future change is tempted to allocate workspaces
lazily: a runtime-allocated Julia object held across C calls is **not reliably rooted** in a
`juliac --trim` library. Measured, twice — `pointer_from_objref` gives a `TypeError` on the
first garbage collection, and an index into a runtime-populated global instead gives a smashed
malloc arena. Both survive a single force evaluation and die in a real run. See the comment
block at the top of the workspace section in `export/src/write_c_interface.jl`.

The LAMMPS plugin allocates `workspaces[t]` for `t < omp_get_max_threads()` in `init_style()`
and passes `workspaces[omp_get_thread_num()]`; `ase_ace.ACELibrary` owns exactly one per
instance.

## Evaluation

```c
double ace_site_energy(void *ws, int z0, int nneigh,
                       const int *neighbor_z, const double *neighbor_Rij);

double ace_site_energy_forces(void *ws, int z0, int nneigh,
                              const int *neighbor_z, const double *neighbor_Rij,
                              double *forces);

double ace_site_energy_forces_virial(void *ws, int z0, int nneigh,
                                     const int *neighbor_z, const double *neighbor_Rij,
                                     double *forces, double *virial);

int    ace_site_basis(void *ws, int z0, int nneigh,
                      const int *neighbor_z, const double *neighbor_Rij,
                      double *basis_out);

void   ace_batch_energy_forces_virial(void *ws, int natoms, const int *z,
                                      const int *neighbor_counts,
                                      const int *neighbor_offsets,
                                      const int *neighbor_z, const double *neighbor_Rij,
                                      double *energies, double *forces, double *virials);
```

* `neighbor_Rij` is `nneigh * 3` doubles, **displacement vectors** `R_j - R_i` in Å, not
  positions.
* `forces` is `nneigh * 3` doubles: the force **on each neighbour**, `-dE_i/dR_j`. The force
  on the centre atom is minus their sum (Newton's third law); LAMMPS does that itself.
* `virial` is 6 doubles in Voigt order `xx, yy, zz, yz, xz, xy`.
* The returned energy is the **site energy including E0** of the centre species (and the pair
  term, when the model has one). `nneigh == 0` returns E0 and writes zeros.
* `ace_site_energy_forces` is the same as the `_virial` entry without the per-edge outer
  product; call it when no virial is wanted.
* `ace_batch_*` is sequential inside one call and uses the one workspace it is given. To
  evaluate concurrently, split the atom range across threads, one workspace each.

## Metadata (no workspace, no state)

```c
double ace_get_cutoff(void);      /* Å */
int    ace_get_n_species(void);
int    ace_get_species(int idx);  /* 1-based; atomic number, or -1 out of range */
int    ace_get_n_basis(void);
unsigned long long ace_build_id(void);   /* provenance; see below */
```

## Diagnostics (not part of the evaluation contract)

```c
long long ace_gc_count(void);     /* collections the library's runtime has performed */
long long ace_alloc_bytes(void);  /* bytes it has allocated */
```

**These are DIAGNOSTIC. Do not build on them.** They are read-only and cost nothing per call,
which is why they are exported unconditionally rather than behind a build flag — a gate whose
precondition depended on how the library was configured would be a worse gate. But their
meaning is Julia's, not this API's: `ace_gc_count` returns `Base.gc_num().pause`, whose exact
semantics belong to the Julia version the library was compiled with and may change under it.
Treat them as "has this number moved?", never as a quantity with a contract.

They exist for one reason. The fault that Task 6's kernel introduced was a workspace reclaimed
by the library's own garbage collector: it passed every single-evaluation accuracy gate and
then died in a real run. So "has this test driven the library hard enough to collect at all?"
is a question a liveness check must be able to *ask*, and no amount of arithmetic about
bytes-per-site answers it reliably — two attempts at that arithmetic were wrong, in opposite
directions. `export/test/python/liveness_gc.py` drives the library until `ace_gc_count()`
reports N collections, requires every result to stay bitwise identical, and fails if the
collections never happen.

Measured on `libace_cantor_poly_b2.so`: a site call at 72–88 neighbours allocates about
`56n + 208` bytes, and the first collection lands at **43.1 MiB** — which is Julia's default
collect interval, `5600 · 1024 · sizeof(void*)` = 43.75 MiB.

## Version and provenance checks

A model compiled before the workspace API exports `ace_site_*` **without** the handle
argument. Calling such a library through the signatures above passes `z0` where the handle
belongs; the handle is tagged and validated, so the call is *rejected* rather than served, but
you still get no result. Both the LAMMPS plugin and `ACELibrary` therefore resolve
`ace_workspace_new` first and **refuse to load** a library that does not export it. Do not
paper over that: re-export and re-compile the model.

`ace_build_id()` returns a 64-bit hash of the generated source the library was compiled from
(everything above the `ACE EXPORT BUILD STAMP` marker line in the `.jl`). Nothing else about a
`.so` records its source, so this is the only way to answer "was this library built from that
model file?" — `export_build_id(<file>)` in `export/src/build_stamp.jl` recomputes it, and
`export/test/runtests.jl` refuses to run the library test groups on a mismatch. Read it out of
process (ctypes, or any C caller): `ccall`ing into a `juliac --trim` library from a host Julia
process aborts that process.

## Cost of a call, and what the entry points are for

Measured through `pair_style ace`, one pinned core, on the two reference models — the full
table, the protocol and the artefacts are in [`../bench/README.md`](../bench/README.md), and
the account of how it got there is in
[`../bench/FINDINGS_parity.md`](../bench/FINDINGS_parity.md).

* `ace_site_energy_forces_virial` is the entry LAMMPS takes on a step that needs the virial;
  `ace_site_energy_forces` is the same computation without the per-edge outer product, and the
  plugin calls it on the steps that do not. The saving is small (~0.25 % of a 100-step run at
  `thermo 50`), and it is measured rather than assumed — a `thermo 1` control run, which forces
  the virial on every step, reproduces the same speed-up figures.
* `ace_site_energy` shares the forward pass and the readout with the force path, so it and
  `ace_site_energy_forces` agree **bitwise** on the many-body term.
* `ace_site_basis` is not on the energy or force path at all. It is served from constants
  (`A2BMAP_*`, `WB_*`) that are emitted **only** when `for_library = true`.
* A site call allocates about `56n + 208` bytes for `n` filtered neighbours. That is real
  allocation in the library's own runtime: it collects, and a caller that drives the library
  hard will see `ace_gc_count()` move. This is not a leak, and the workspace is not affected —
  the pool is part of the image precisely so that a collection cannot reclaim it.

**Do not size a workload from these numbers** without re-reading the protocol: they are
single-core, `timestep 0.0`, and include the plugin's per-site neighbour copy.
