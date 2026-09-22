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

OUTPUT FORMAT.  `.yace` (YAML).  The acejax original wrote the TEXT C-tilde format with
`.save()`, which is fine for one element but writes a single `radbasename=` line that is
invalid for more than one -- this LAMMPS build AND pyace's own reader both reject it with
"`radbasename` array has wrong shape. It must be of shape (nelements, nelements)".
`save_yaml()` writes the per-ordered-pair form; the script reloads what it wrote and asserts
the basis size survived the round trip.

    /storage/eng/essswb/venvs/pyacenv/bin/python make_pace_basis.py \
        --target 2369 --order 4 --lmax 7 --rcut 5.5 --elements Ti,Al \
        --nradmax-by-orders 22,6,2,1 --out bench_parity/tial_o4_pace.yace
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


def randomise(bc):
    """Give every basis function a random coefficient.

    NOT `block.set_all_coeffs(...)`, which the single-element original used: that also
    rewrites the block's radial coefficients, and for a multi-species configuration the
    (i,j) and (j,i) blocks then disagree, so `ACEBBasisSet(bc)` raises
    "Bonds specifications for pair (1,0) are inconsistent".  Writing `funcspec.coeffs`
    leaves the bond/radial specification untouched and converts cleanly.  Verified: the
    B-basis and C-tilde function counts are identical before and after.
    """
    rng = np.random.default_rng(0)
    for blk in bc.funcspecs_blocks:
        for f in blk.funcspecs:
            f.coeffs = rng.normal(scale=0.01, size=len(f.coeffs)).tolist()
    return bc


def ctilde_per_element(bc):
    """(C-tilde basis set, functions per central element).

    This -- not the B-basis count -- is what `pair_style pace` actually evaluates per
    atom, so it is what the parity benchmark matches on.  The two differ by ~10% here.
    """
    ct = ACEBBasisSet(bc).to_ACECTildeBasisSet()
    per = [len(r1) + len(rn) for r1, rn in zip(ct.basis_rank1, ct.basis)]
    return ct, per


if EXPLICIT is not None:
    assert len(EXPLICIT) == args.order, \
        f"--nradmax-by-orders needs {args.order} entries, got {len(EXPLICIT)}"
    candidates = [tuple(EXPLICIT)]
else:
    # A uniform nradmax scan (what the single-element original did) is far too coarse
    # here: for Ti-Al order 4 it steps 1526 -> 17414 -> 78760 total functions.  Scan the
    # per-order radial cutoffs instead, which is the knob that actually resolves a few
    # thousand functions.  Stage 1 ranks the grid by the (cheap) B-basis count; stage 2
    # converts only the best few and picks on the C-tilde count.
    if args.order == 4:
        grid = [(n1, n2, n3, n4)
                for n1 in (8, 12, 16, 20, 22, 26)
                for n2 in range(1, 9)
                for n3 in range(1, 5)
                for n4 in range(1, 4)
                if n2 >= n3 >= n4]
    else:
        grid = [tuple([nr] * args.order) for nr in range(1, 20)]
    scored = []
    for nrb in grid:
        try:
            _, n = build(nrb)
        except Exception as exc:                     # noqa: BLE001 - pyace raises bare errors
            print(f"  {nrb}: {exc}", file=sys.stderr)
            continue
        scored.append((abs(n - TOTAL_TARGET), n, nrb))
    scored.sort()
    for d, n, nrb in scored[:10]:
        print(f"  B-basis  nradmax_by_orders={nrb}: {n} total ({n / NEL:.0f} per species)",
              file=sys.stderr)
    candidates = [nrb for _, _, nrb in scored[:6]]

best = None
for nrb in candidates:
    bc, n = build(nrb)
    _, per = ctilde_per_element(bc)
    print(f"  C-tilde  nradmax_by_orders={nrb}: {per} per element "
          f"(B-basis {n} total)", file=sys.stderr)
    score = abs(max(per) - args.target)
    if best is None or score < best[0]:
        best = (score, nrb, n, per)

_, nradmax, n, per = best
bc, _ = build(nradmax)
print(f"target {args.target} many-body functions per central species "
      f"(ACEpotentials `length(model.tensor)`)")
print(f"  chosen nradmax_by_orders={nradmax}, order={args.order}, lmax={args.lmax}, "
      f"rcut={args.rcut}, elements={','.join(ELEMENTS)}")
print(f"  B-basis  : {n} total = {n / NEL:.1f} per species "
      f"({100.0 * (n / NEL - args.target) / args.target:+.1f}%)")
print(f"  C-tilde  : {per} per element   <-- what pair_style pace evaluates "
      f"({100.0 * (max(per) - args.target) / args.target:+.1f}%)")
print("  B-basis per-block counts: "
      + ", ".join(f"{b.block_name}={len(b.funcspecs)}" for b in bc.funcspecs_blocks))

# pair_style pace reads the C-TILDE basis, not the B-basis configuration that
# BBasisConfiguration.save() writes -- converting is the whole point of this step.
ct, per_after = ctilde_per_element(randomise(bc))
assert per_after == per, f"randomising coefficients changed the basis size: {per} -> {per_after}"

# `.save()` (the TEXT C-tilde format the single-element original used) is BROKEN for more
# than one element in python-ace 0.2.8: it writes a single `radbasename=ChebExpCos` line,
# and both this LAMMPS build and pyace's own reader then reject the file with
#   ValueError: `radbasename` array has wrong shape. It must be of shape (nelements, nelements)
# `save_yaml()` writes the per-ordered-pair form, which loads.  So the output is YAML (.yace)
# regardless of what the original docstring said about `.ace`.
assert args.out.endswith((".yace", ".yaml", ".yml")), \
    f"--out must be a .yace file (YAML C-tilde format), got {args.out}"
ct.save_yaml(args.out)
from pyace import ACECTildeBasisSet                               # noqa: E402
reloaded = ACECTildeBasisSet(args.out)
per_reload = [len(r1) + len(rn) for r1, rn in zip(reloaded.basis_rank1, reloaded.basis)]
assert per_reload == per, f"reloaded basis size differs: {per} -> {per_reload}"
print(f"wrote {args.out} (reloaded OK, {per_reload} C-tilde functions per element)")
