#!/usr/bin/env python3
"""liveness_gc.py -- drive a compiled ACE library until its runtime has COLLECTED, and check
the answer is still bit-identical.

WHY THIS EXISTS.  The Task 6 fault was a workspace reclaimed by the library's own garbage
collector: it passed every `run 0` accuracy gate in the plan and then died in a real run.  The
property a liveness check has to establish is therefore not "many steps ran" but "the library
garbage-collected, with a live workspace in hand, and kept answering correctly".

WHY IT ASSERTS THE GC COUNT INSTEAD OF SIZING A WORKLOAD.  Sizing by arithmetic was tried and
got the wrong answer twice: once from LAMMPS' `Ave neighs/atom`, which is the UNFILTERED
neighbour list including the 2 A skin (201 rather than the 82.5 the plugin passes in on the
Cantor benchmark box), and once from reading the abort banner's allocation counter.  The
library now exports `ace_gc_count()` and `ace_alloc_bytes()`, so the condition can be observed
instead of predicted.  Measured on `libace_cantor_poly_b2.so`: the first collection lands at
**43.1 MB** allocated, i.e. 8 796 site calls at 82-88 neighbours -- under a second of driving.

WHAT IT ASSERTS
  1. `ace_gc_count()` advances by at least `--gcs` (default 3).  If the cap is reached first,
     it FAILS: a run that never collected proves nothing, exactly as a `-DBUILD_OMP=OFF`
     plugin proves nothing in the OpenMP comparison.
  2. Every energy, force and virial component stays BITWISE equal to the first call's.  Not
     `close`: a workspace reclaimed and reused produces results that are usually right and
     occasionally wrong, and only an exact comparison catches the occasional one reliably.
  3. The workspace handle is still accepted at the end (it was not invalidated by a GC).

    liveness_gc.py <libace.so> [--gcs N] [--neigh N] [--max-calls N] [--max-seconds S]

WHAT IT DOES NOT COVER, because a gate is more useful when its blind spots are written down.
It calls only `ace_site_energy_forces_virial`, with ONE workspace, ONE geometry and a FIXED
`n = 72`.  A lifetime fault confined to `ace_site_energy`, `ace_site_basis` or `ace_batch_*`,
one that needs two workspaces live at once, or one that only appears at a neighbour count in a
different allocation size class, would not be seen here.  Gates A/B/C and the LAMMPS NVE run
(L2) cover the other entry points and a real trajectory; the OpenMP comparison covers
concurrent workspaces.

Prints one `LIVENESS_GC PASS` / `FAIL` line last, greppable by a caller that wants a verdict.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--gcs", type=int, default=3, help="collections to require (default 3)")
    ap.add_argument("--neigh", type=int, default=72, help="neighbours per synthetic site")
    ap.add_argument("--max-calls", type=int, default=20_000_000)
    # A wall-clock bound as well as a call bound.  Without it a library that never collects
    # spends 15-20 minutes reaching --max-calls before failing; the whole point of this gate
    # is that it answers in seconds (3 collections take 1.7 s on the Cantor library).
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--label", default="LIVENESS_GC")
    a = ap.parse_args(argv)

    lib = ctypes.CDLL(a.library)

    # Resolve EVERY symbol first, and report a missing one as a verdict rather than as a
    # traceback.  The first version bound the required symbols in a bare loop and only
    # guarded ace_gc_count, so a pre-workspace library raised
    # `AttributeError: undefined symbol: ace_workspace_new` before the message explaining
    # what to do could ever print.  It failed closed, but unhelpfully.
    wanted = {
        "ace_workspace_new": (ctypes.c_void_p, []),
        "ace_workspace_free": (None, [ctypes.c_void_p]),
        "ace_get_cutoff": (ctypes.c_double, []),
        "ace_get_n_species": (ctypes.c_int, []),
        "ace_get_species": (ctypes.c_int, [ctypes.c_int]),
        "ace_gc_count": (ctypes.c_longlong, []),
        "ace_alloc_bytes": (ctypes.c_longlong, []),
        "ace_site_energy_forces_virial": (ctypes.c_double, [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]),
    }
    missing = []
    for name, (res, args) in wanted.items():
        try:
            f = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        f.restype, f.argtypes = res, args
    if missing:
        print(f"{a.label}: {a.library} exports none of: {', '.join(missing)}.  A library that "
              f"predates the workspace API or the liveness diagnostic cannot be checked by "
              f"this gate -- re-export and re-compile it with the current "
              f"export_ace_model.jl.")
        print(f"{a.label} FAIL")
        return 2

    ws = lib.ace_workspace_new()
    if not ws:
        print(f"{a.label}: ace_workspace_new() returned NULL")
        print(f"{a.label} FAIL")
        return 2

    rcut = lib.ace_get_cutoff()
    species = [lib.ace_get_species(i + 1) for i in range(lib.ace_get_n_species())]

    # A synthetic site: `neigh` neighbours on random directions at radii spread over
    # [0.4 rcut, 0.98 rcut), species drawn round-robin.  The geometry does not have to be
    # physical -- what is being tested is the library's lifetime behaviour, and the bitwise
    # comparison is against this same site's own first answer.
    rng = np.random.default_rng(20260917)
    n = a.neigh
    d = rng.random((n, 3)) - 0.5
    d /= np.linalg.norm(d, axis=1)[:, None]
    d *= (0.40 * rcut + 0.58 * rcut * rng.random(n))[:, None]
    R = np.ascontiguousarray(d.flatten(), dtype=np.float64)
    Z = np.ascontiguousarray(np.array([species[i % len(species)] for i in range(n)],
                                      dtype=np.int32))
    z0 = species[0]
    F = np.zeros(n * 3, dtype=np.float64)
    V = np.zeros(6, dtype=np.float64)

    pR = R.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pZ = Z.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    pF = F.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pV = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    hws = ctypes.c_void_p(ws)

    def call():
        return lib.ace_site_energy_forces_virial(hws, z0, n, pZ, pR, pF, pV)

    E0 = call()
    F0, V0 = F.copy(), V.copy()
    if not np.isfinite(E0):
        print(f"{a.label}: first call returned {E0}")
        print(f"{a.label} FAIL")
        return 1

    gc0 = lib.ace_gc_count()
    b0 = lib.ace_alloc_bytes()
    t0 = time.time()
    calls = 0
    bad = 0
    while (lib.ace_gc_count() - gc0 < a.gcs and calls < a.max_calls
           and time.time() - t0 < a.max_seconds):
        E = call()
        calls += 1
        if E != E0 or not np.array_equal(F, F0) or not np.array_equal(V, V0):
            bad += 1
            if bad == 1:
                print(f"{a.label}: FIRST MISMATCH at call {calls}: dE = {E - E0!r}, "
                      f"max|dF| = {np.max(np.abs(F - F0))!r}, "
                      f"max|dV| = {np.max(np.abs(V - V0))!r}")
    gcs = lib.ace_gc_count() - gc0
    mb = (lib.ace_alloc_bytes() - b0) / 2 ** 20
    dt = time.time() - t0

    print(f"{a.label}: {calls} calls, {n} neighbours, {mb:.1f} MB allocated, "
          f"{gcs} collection(s), {dt:.2f} s")
    ok = True
    if gcs < a.gcs:
        why = ("hit --max-seconds" if dt >= a.max_seconds else
               "hit --max-calls" if calls >= a.max_calls else "loop ended early")
        print(f"{a.label}: the library collected {gcs} time(s) in {calls} calls ({why}) -- "
              f"asked for {a.gcs}.  A run that never collected CANNOT see the fault this "
              f"gate exists for, so this is a FAILURE of the check, not a pass.")
        ok = False
    if bad:
        print(f"{a.label}: {bad} of {calls} calls disagreed BITWISE with the first -- "
              f"the workspace is not surviving collection")
        ok = False

    # The handle must still be live after all that.
    if call() != E0:
        print(f"{a.label}: the workspace handle no longer reproduces the first answer")
        ok = False
    lib.ace_workspace_free(hws)

    print(f"{a.label} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
