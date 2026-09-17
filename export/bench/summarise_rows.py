#!/usr/bin/env python3
"""summarise_rows.py <rows file> [tag-prefix ...] -- the POOLED-RUN MEDIAN of a set of rows.

Exists because a figure in export/bench/README.md was once written by hand into the very
paragraph whose point was that the figures are computed from the artefact.  Three of the four
entries matched a pooled-run median of their rows; the fourth (`tial_poly`, 182.28) matched no
aggregation of anything and was simply mistyped.  Quoting a number from a rows file now means
running this and pasting what it prints.

The statistic is the SAMPLE MEDIAN over every run of every named block, pooled -- not the
median of the block medians, and not a mean.  `bench_parity.sh` reports a per-block median
because a block is two or three runs taken together; when several blocks of the same binary
are combined, pooling their runs is what the +-7 % block-to-block scatter Task 4 documented
calls for.

    export/bench/summarise_rows.py bench_parity/rows_task6_fix3.txt cantor_poly tial_poly
"""
import re
import sys

ROW = re.compile(r"^(\S+)\s.*natoms=(\d+).*runs\(exec order\)=\s*([-\d. ]+?),\s*spread=([\d.]+)")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    rows = []
    for line in open(argv[1]):
        m = ROW.match(line)
        if not m:
            continue
        tag, nat, runs, spread = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
        vals = [float(x) for x in runs.split() if x != "-"]
        rows.append((tag, nat, vals, spread))
    wanted = argv[2:]
    groups = {}
    for tag, nat, vals, spread in rows:
        key = next((w for w in wanted if tag.startswith(w)), None) if wanted else tag
        if key is None:
            continue
        groups.setdefault(key, {"nat": nat, "runs": [], "blocks": []})
        groups[key]["runs"] += vals
        groups[key]["blocks"].append((tag, vals, spread))
    for key, g in groups.items():
        v = sorted(g["runs"])
        n = len(v)
        med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
        print(f"{key}: {len(g['blocks'])} block(s), {n} run(s), pooled median = {med:.4f} "
              f"ms/step = {med * 1000 / g['nat']:.1f} us/site")
        for tag, vals, spread in g["blocks"]:
            flag = "  <-- INVALID BLOCK (> 3 % internal spread)" if spread > 0.03 else ""
            print(f"    {tag:<28} {' '.join(f'{x:.3f}' for x in vals)}"
                  f"   spread={spread * 100:.2f} %{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
