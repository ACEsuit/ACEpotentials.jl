#!/usr/bin/env python3
"""summarise_rows.py <rows file> [tag-prefix ...] -- the POOLED-RUN MEDIAN of a set of rows.

    export/bench/summarise_rows.py bench_parity/rows_task6_fix3.txt cantor_poly tial_poly

WHAT THE STATISTIC IS.  The sample median over every run of every INCLUDED block, pooled --
not the median of the block medians, and not a mean.  `bench_parity.sh` reports a per-block
median because a block is two or three runs taken together; when several blocks of the same
binary are combined, pooling their runs is what the +-7 % block-to-block scatter Task 4
documented calls for.

WHAT IT EXCLUDES, AND WHAT IT DOES NOT -- read this before quoting a number:

  * EXCLUDED automatically: any block whose own runs disagree by more than `--max-spread`
    (default 3 %).  That is the protocol's rule in `README.md` -- two runs must agree within
    3 %, and a block that fails it was taken on a contended core and is not a measurement.
  * EXCLUDED explicitly: any block named by a `# EXCLUDE <tag> <reason>` line in the ROWS FILE
    itself, or by `--exclude <tag>=<reason>` on the command line.  A reason is REQUIRED.  This
    is how a block is dropped for something the spread rule does not capture -- an outlier
    whose internal spread is fine.
  * NOT excluded by anything else.  There is no outlier rejection, no trimming, no "drop the
    first block", no smoothing.  A number this tool prints is the median of every run it was
    given minus the two categories above, both of which it lists by name in its own output.

WHY IT EXISTS, AND WHY IT IS BUILT THIS WAY.  A control figure in `export/bench/README.md`
(`tial_poly` 182.28) was written by hand instead of computed, in the paragraph whose point was
that these figures come from the artefact.  This tool was then added to make that impossible --
and its FIRST version had the same defect in a subtler form: it printed the >3 % flag and then
pooled the flagged block anyway, so run as documented it produced 116.9760 / 127.6610 /
182.3600 against a README saying 116.8575 / 128.4955 / 182.1295, and reproduced the table only
if the caller already knew to type block-specific prefixes that dodged the unwanted block by
name.  That is the manual judgement call the tool exists to remove, and it would not have
caught the original error either -- it produced a third wrong number.  Hence: the exclusions
are applied, not printed; they are named in the output; and an exclusion for a reason the
spread rule does not capture has to be WRITTEN DOWN, in the rows file or the invocation, where
the next reader will see it.

IT FAILS LOUDLY.  A malformed file, an empty file, a tag matching no rows, a `# EXCLUDE` line
naming a tag that is not there, or a group left with no included blocks are all errors with a
message and a nonzero exit.  A quoting tool that prints nothing when the format drifts is
worse than no tool, because the silence looks like a clean run.
"""
import argparse
import re
import sys

# The ACE half of the row, anchored on `ace_us/site=`.
#
# THE ANCHOR IS THE WHOLE POINT, and it was missing.  A row that carries a `pace` comparator
# ends `... ace_us/site=X (n=2 runs(exec order)= A B, spread=S)  pace_recursive_ms/step(...)=Y
# pace_us/site=Z (n=3 runs(exec order)= C D E, spread=T)`, and the original pattern's greedy
# `.*runs\(exec order\)=` matched the LAST one -- so on any row with a comparator this tool
# reported the ML-PACE runs as if they were the exported code's.  Nothing caught it because
# `rows_task6_fix3.txt`, the only file the tool was validated against, was taken with
# `pace=none` on every row; `rows_task5.txt`, `rows_task6.txt` and `rows_task6_full.txt` all
# contain rows it would have mis-read, and it had never been run on them.  Found by Task 7,
# whose first block was `cantor_poly_b3` WITH the comparator and came back reading 132 ms/step
# (the comparator) against a row saying 155.98.
#
# Requiring `ace_us/site=` rather than merely making the wildcard lazy means a future row
# format that moves or renames the ACE half makes this tool FAIL LOUDLY ("no rows matched")
# instead of quietly picking whichever `runs(...)` it finds first.  Every row that has ever
# carried `runs(exec order)` also carries `ace_us/site=`; Task 4's pre-fix rows carry neither
# and were already not matched.
ROW = re.compile(r"^(\S+)\s.*?natoms=(\d+).*?"
                 r"\bace_us/site=[\d.]+\s*\(n=\d+\s+runs\(exec order\)=\s*"
                 r"([-\d. ]+?),\s*spread=([\d.]+)\)")
EXCL = re.compile(r"^#\s*EXCLUDE\s+(\S+)\s+(.*\S)\s*$")


def die(msg):
    print(f"summarise_rows.py: {msg}", file=sys.stderr)
    return 2


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows_file")
    ap.add_argument("prefixes", nargs="*",
                    help="tag prefixes to group by; default is one group per exact tag")
    ap.add_argument("--max-spread", type=float, default=0.03,
                    help="a block whose internal spread exceeds this is excluded (default 0.03)")
    ap.add_argument("--exclude", action="append", default=[], metavar="TAG=REASON",
                    help="exclude one block by exact tag; a reason is required")
    a = ap.parse_args(argv)

    try:
        text = open(a.rows_file).read()
    except OSError as e:
        return die(f"cannot read {a.rows_file}: {e}")

    excluded = {}
    for spec in a.exclude:
        if "=" not in spec:
            return die(f"--exclude {spec!r} has no reason; use --exclude TAG=REASON")
        tag, reason = spec.split("=", 1)
        if not reason.strip():
            return die(f"--exclude {spec!r} has an empty reason")
        excluded[tag] = reason.strip()

    rows = []
    for line in text.splitlines():
        m = EXCL.match(line)
        if m:
            excluded.setdefault(m.group(1), m.group(2))
            continue
        m = ROW.match(line)
        if m:
            rows.append((m.group(1), int(m.group(2)),
                         [float(x) for x in m.group(3).split() if x != "-"],
                         float(m.group(4))))

    if not rows:
        return die(f"{a.rows_file}: no rows matched the bench_parity.sh row format "
                   f"({len(text.splitlines())} line(s) read). Has the row format changed?")

    tags = {t for t, _, _, _ in rows}
    for tag in excluded:
        if tag not in tags:
            return die(f"exclusion names tag {tag!r}, which is not in {a.rows_file}. "
                       f"A stale exclusion silently changes a statistic; fix or remove it.")

    wanted = a.prefixes
    for w in wanted:
        if not any(t.startswith(w) for t in tags):
            return die(f"no row in {a.rows_file} has a tag starting with {w!r}")

    groups = {}
    order = []
    for tag, nat, vals, spread in rows:
        key = next((w for w in wanted if tag.startswith(w)), None) if wanted else tag
        if key is None:
            continue
        if key not in groups:
            groups[key] = {"nat": nat, "runs": [], "blocks": []}
            order.append(key)
        if tag in excluded:
            why = f"EXCLUDED: {excluded[tag]}"
        elif spread > a.max_spread:
            why = (f"EXCLUDED: internal spread {spread * 100:.2f} % > "
                   f"{a.max_spread * 100:.0f} % (protocol)")
        else:
            why = "included"
            groups[key]["runs"] += vals
        groups[key]["blocks"].append((tag, vals, spread, why))

    rc = 0
    for key in order:
        g = groups[key]
        v = sorted(g["runs"])
        n = len(v)
        if n == 0:
            print(f"{key}: EVERY block excluded -- no statistic")
            for tag, vals, spread, why in g["blocks"]:
                print(f"    {tag:<28} {' '.join(f'{x:.3f}' for x in vals)}   {why}")
            rc = 1
            continue
        med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
        nin = sum(1 for b in g["blocks"] if b[3] == "included")
        print(f"{key}: {nin} of {len(g['blocks'])} block(s) included, {n} run(s), "
              f"pooled median = {med:.4f} ms/step = {med * 1000 / g['nat']:.1f} us/site")
        for tag, vals, spread, why in g["blocks"]:
            print(f"    {tag:<28} {' '.join(f'{x:.3f}' for x in vals)}"
                  f"   spread={spread * 100:.2f} %   {why}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
