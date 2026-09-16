#!/usr/bin/env python3
"""compare_dump.py <dump> <ref> <tol> [options] -- gate two force sets against each other.

Generalisation of ``verify_cantor/compare_lammps.py``, which could only compare a
``dump.<tag>.<k>`` against the Cantor ``ref_<k>.txt`` columns.  This one takes any two files
and one tolerance, so it can serve every pairing the plan needs:

    1 rank vs 2 ranks          compare_dump.py dump.np2 dump.np1 1e-12
    OMP_NUM_THREADS=4 vs 1     compare_dump.py dump.omp4 dump.omp1 1e-12      (Task 6)
    a new generator vs the old compare_dump.py dump.new dump.old 1e-13        (Tasks 5-7)
    LAMMPS vs a 17-digit ref   compare_dump.py dump.ace ref_3.txt 1e-10 --ref-cols 7

Both arguments may be a LAMMPS dump (any column order -- the ``ITEM: ATOMS`` header is
parsed) or a plain whitespace table whose first column is the atom id.  For a plain table
``--ref-cols C`` names the 1-based column of ``fx`` (``fy``/``fz`` follow it); the default,
2, reads ``id fx fy fz``.  ``verify_cantor/ref_<k>.txt`` holds five force blocks, so its
columns are 2, 5, 8, 11, 14 for a/amb/bmb/c50/c200.

Atoms are matched by id, never by file order, so an MPI run that dumps atoms in a different
order than a serial run still compares correctly.

The gated quantity is ``max_i || F_i - Fref_i ||`` in eV/Å -- a force is intensive, so it is
compared ABSOLUTELY and never divided by the number of atoms.  Positions are compared too
when both files carry them, but only REPORTED: a position mismatch means the two runs did
not evaluate the same geometry, which the caller should treat as a setup error.

``--energy E Eref`` additionally gates ``|E - Eref| / natoms`` in eV/atom at the same
tolerance (energy is extensive, hence per atom -- the same convention as
``export/test/check_export.jl``).

Exit status 0 = PASS, 1 = FAIL, 2 = could not compare.  The last line is always
``<label> PASS`` or ``<label> FAIL``, greppable by a caller that only wants the verdict.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _read_dump(path: str, force_col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (ids, forces, positions-or-None) from a LAMMPS dump or a plain table."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    header = next((i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS")), None)

    if header is not None:
        cols = lines[header].split()[2:]
        try:
            i_id = cols.index("id")
        except ValueError:
            raise SystemExit(f"{path}: dump has no `id` column (columns: {cols})")
        if "fx" not in cols:
            raise SystemExit(f"{path}: dump has no `fx` column (columns: {cols})")
        i_f = cols.index("fx")
        i_x = cols.index("x") if "x" in cols else None
        rows = np.array([[float(x) for x in l.split()] for l in lines[header + 1:] if l.strip()])
    else:
        rows = np.array(
            [[float(x) for x in l.split()] for l in lines if l.strip() and not l.lstrip().startswith("#")]
        )
        if rows.ndim != 2:
            raise SystemExit(f"{path}: not a rectangular numeric table")
        i_id, i_f, i_x = 0, force_col - 1, None

    if rows.size == 0:
        raise SystemExit(f"{path}: no atom rows")
    if i_f + 3 > rows.shape[1]:
        raise SystemExit(f"{path}: only {rows.shape[1]} columns, cannot read forces at column {i_f + 1}")

    order = np.argsort(rows[:, i_id])
    rows = rows[order]
    ids = rows[:, i_id].astype(np.int64)
    F = rows[:, i_f:i_f + 3]
    X = rows[:, i_x:i_x + 3] if i_x is not None else None
    return ids, F, X


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare the forces in two LAMMPS dumps.")
    p.add_argument("dump")
    p.add_argument("ref")
    p.add_argument("tol", type=float)
    p.add_argument("--ref-cols", type=int, default=2,
                   help="1-based column of fx in a plain-table reference (default 2)")
    p.add_argument("--dump-cols", type=int, default=2,
                   help="1-based column of fx in a plain-table dump (default 2)")
    p.add_argument("--energy", nargs=2, type=float, metavar=("E", "EREF"),
                   help="also gate |E - EREF| / natoms at the same tolerance")
    p.add_argument("--label", default="COMPARE_DUMP", help="prefix of the verdict line")
    args = p.parse_args(argv)

    ids_a, F_a, X_a = _read_dump(args.dump, args.dump_cols)
    ids_b, F_b, X_b = _read_dump(args.ref, args.ref_cols)

    if ids_a.shape != ids_b.shape or not np.array_equal(ids_a, ids_b):
        print(f"{args.dump}: {ids_a.size} atoms, {args.ref}: {ids_b.size} atoms -- id sets differ")
        print(f"{args.label} FAIL")
        return 2

    d = np.linalg.norm(F_a - F_b, axis=1)
    fmax = float(np.max(np.linalg.norm(F_a, axis=1)))
    ok = bool(d.max() <= args.tol)

    print(f"{args.label}: {ids_a.size} atoms   max|F| = {fmax:.6e} eV/A")
    print(f"{args.label}: max|dF| = {d.max():.6e} eV/A   mean|dF| = {d.mean():.6e}   "
          f"tol = {args.tol:.1e}   [absolute, per atom vector norm]")
    if X_a is not None and X_b is not None:
        dx = float(np.max(np.abs(X_a - X_b)))
        print(f"{args.label}: max|dx| = {dx:.6e} A   [reported only -- a non-zero value means "
              f"the two runs did not evaluate the same geometry]")

    if args.energy is not None:
        e, eref = args.energy
        de = abs(e - eref) / ids_a.size
        print(f"{args.label}: |dE|/atom = {de:.6e} eV/atom   tol = {args.tol:.1e}   [per atom]")
        ok = ok and bool(de <= args.tol)

    print(f"{args.label} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
