"""Build a `pair_style pace` basis matched to an ACEpotentials model's SIZE.

Adapted from ~/si-ace/ACEpotentials/acejax/bench/make_pace_basis.py (single-element)
to take an arbitrary element list, because the parity benchmark's large reference
model is two-element Ti-Al.

WHY RANDOM COEFFICIENTS ARE FINE.  For a throughput benchmark the coefficients are
irrelevant: pace evaluates every basis function regardless of their values, and with
`timestep 0.0` the atoms never move, so a physically meaningless potential cannot
destabilise anything.  This is a COST comparison at matched basis size, not a
comparison of two fits.

NOTE ON THE TWO COMPARATORS IN THIS PLAN.  The Cantor comparator
(~/si-ace/spike_yace/cantor/cantor_n10000_exact.yace) is NOT built this way -- it is
the *same physical model* exported exactly from an ACEpotentials v0.6 fit by
spike_yace/cantor_all.jl.  The TiAl comparator built here is only size-matched.  Both
facts are recorded next to the rows in README.md.

BASIS COUNTING.  `--target` is the number of many-body basis functions **per central
species** -- the quantity ACEpotentials reports as `size(ps.WB, 1)` / `length(model.tensor)`.
pyace's natural count is the total over all central species, so for N elements the
script scans for `N * target` total functions.  Both numbers are printed.

ENVIRONMENT.  python-ace ships cp39 wheels only and imports pkg_resources:

    uv venv --python 3.9 /storage/eng/essswb/venvs/pyacenv
    uv pip install --python /storage/eng/essswb/venvs/pyacenv/bin/python python-ace "setuptools<81"

Write the output as `.ace`, NOT `.yace`: `to_ACECTildeBasisSet().save()` emits the TEXT
format and a `.yace` extension makes yaml-cpp reject it.

    /storage/eng/essswb/venvs/pyacenv/bin/python make_pace_basis.py \
        --target 2369 --order 4 --lmax 7 --rcut 5.5 --elements Ti,Al \
        --out bench_parity/tial_o4_pace.ace
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from pyace import ACEBBasisSet, create_multispecies_basis_config

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, required=True,
               help="many-body basis functions PER CENTRAL SPECIES to match")
p.add_argument("--order", type=int, required=True, help="correlation order (body order - 1)")
p.add_argument("--lmax", type=int, required=True)
p.add_argument("--rcut", type=float, required=True)
p.add_argument("--elements", type=str, required=True, help="comma separated, e.g. Ti,Al")
p.add_argument("--out", type=str, required=True)
p.add_argument("--nradmax-by-orders", type=str, default=None,
               help="explicit comma-separated nradmax per order; skips the scan")
p.add_argument("--ndens", type=int, default=1)
args = p.parse_args()

ELEMENTS = [e.strip() for e in args.elements.split(",") if e.strip()]
NEL = len(ELEMENTS)
TOTAL_TARGET = args.target * NEL
EXPLICIT = ([int(x) for x in args.nradmax_by_orders.split(",")]
            if args.nradmax_by_orders else None)


# Multi-species note: `"bonds": {"ALL": {...}}` builds a configuration whose function
# count is fine but which `ACEBBasisSet(bc)` then rejects with
#   ValueError: Bonds specifications for pair (1,0) are inconsistent
# The per-ordered-pair spelling below converts cleanly.  (The single-element original
# never hit this.)
BONDS = {(a, b): {"radbase": "ChebExpCos", "radparameters": [5.25], "rcut": args.rcut,
                  "dcut": 0.01, "NameOfCutoffFunction": "cos"}
         for a in ELEMENTS for b in ELEMENTS}


def build(nradmax_by_orders):
    cfg = {
        "deltaSplineBins": 0.001,
        "elements": ELEMENTS,
        "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                               "fs_parameters": [1, 1], "ndensity": args.ndens,
                               "rho_core_cut": 100000, "drho_core_cut": 250}},
        "bonds": BONDS,
        "functions": {"ALL": {"nradmax_by_orders": list(nradmax_by_orders),
                              "lmax_by_orders": [0] + [args.lmax] * (args.order - 1)}},
    }
    bc = create_multispecies_basis_config(cfg)
    n = sum(len(b.funcspecs) for b in bc.funcspecs_blocks)
    return bc, n


if EXPLICIT is not None:
    assert len(EXPLICIT) == args.order, \
        f"--nradmax-by-orders needs {args.order} entries, got {len(EXPLICIT)}"
    bc, n = build(EXPLICIT)
    best = (bc, tuple(EXPLICIT), n)
else:
    # A uniform nradmax scan (what the single-element original did) is far too coarse
    # here: for Ti-Al order 4 it steps 1526 -> 17414 -> 78760 total functions.  Scan the
    # per-order radial cutoffs instead, which is the knob that actually resolves a few
    # thousand functions.
    best = None
    grid = [(n1, n2, n3, n4)
            for n1 in (8, 12, 16, 20, 22, 26)
            for n2 in range(1, 9)
            for n3 in range(1, 5)
            for n4 in range(1, 4)
            if n2 >= n3 >= n4][:1000] if args.order == 4 else None
    if grid is None:
        grid = [tuple([nr] * args.order) for nr in range(1, 20)]
    for nrb in grid:
        try:
            bc, n = build(nrb)
        except Exception as exc:                     # noqa: BLE001 - pyace raises bare errors
            print(f"  {nrb}: {exc}", file=sys.stderr)
            continue
        print(f"  nradmax_by_orders={nrb}: {n} total ({n / NEL:.0f} per species)",
              file=sys.stderr)
        if best is None or abs(n - TOTAL_TARGET) < abs(best[2] - TOTAL_TARGET):
            best = (bc, nrb, n)

bc, nradmax, n = best
print(f"target {args.target} functions/species ({TOTAL_TARGET} total for {NEL} elements) "
      f"-> achieved {n} total = {n / NEL:.1f} per species "
      f"(nradmax_by_orders={nradmax}, order={args.order}, lmax={args.lmax}, "
      f"rcut={args.rcut}, elements={','.join(ELEMENTS)}); "
      f"mismatch {100.0 * (n - TOTAL_TARGET) / TOTAL_TARGET:+.1f}%")
print("per-block function counts: "
      + ", ".join(f"{b.block_name}={len(b.funcspecs)}" for b in bc.funcspecs_blocks))

rng = np.random.default_rng(0)
for b in bc.funcspecs_blocks:
    b.set_all_coeffs(rng.normal(scale=0.01, size=len(b.get_all_coeffs())).tolist())

# pair_style pace reads the C-TILDE basis, not the B-basis configuration that
# BBasisConfiguration.save() writes -- converting is the whole point of this step.
ACEBBasisSet(bc).to_ACECTildeBasisSet().save(args.out)
print("wrote", args.out)
