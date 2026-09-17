#!/usr/bin/env python3
"""summarise_rows.py <rows file> [tag-prefix ...] -- the POOLED-RUN MEDIAN of a set of rows.

    export/bench/summarise_rows.py bench_parity/rows_task6_fix3.txt cantor_poly tial_poly
    export/bench/summarise_rows.py bench_parity/rows_task8.txt --series both

`--series` picks which pair style is summarised: `ace` (the default, and what every earlier
task quoted), `pace` for the `pair_style pace recursive` comparator, or `both`, which prints
the two pooled medians and their ratio.  A published RATIO must come from `both`: the
numerator and the denominator are then pooled over the same included blocks by the same rule,
whereas an ACE median from here divided by a pace median read off a row by eye is exactly the
half-tool, half-hand arithmetic this script exists to prevent.

WHAT THE STATISTIC IS.  The sample median over every run of every INCLUDED block, pooled --
not the median of the block medians, and not a mean.  `bench_parity.sh` reports a per-block
median because a block is two or three runs taken together; when several blocks of the same
binary are combined, pooling their runs is what the +-7 % block-to-block scatter Task 4
documented calls for.

WHAT IT EXCLUDES, AND WHAT IT DOES NOT -- read this before quoting a number:

  * EVERY BLOCK COUNTS unless it is explicitly excluded.  `bench_parity.sh` takes two runs and,
    only if they disagree by more than 3 %, a third; a 2-run block contributes both runs, a
    3-run block contributes ONE value -- its MEDIAN, which is what the protocol says that block
    measured.
  * EXCLUDED automatically: only a 2-run block whose runs disagree by more than `--max-spread`
    AND which has no third run, i.e. a block the protocol's escalation was never applied to.
    `bench_parity.sh` cannot produce one; a hand-edited or truncated rows file can.
  * EXCLUDED explicitly: any block named by a `# EXCLUDE <key> <reason>` line in the ROWS FILE
    itself, or by `--exclude <key>=<reason>`.  A reason is REQUIRED.  `<key>` is `TAG` (every
    block of that tag) or `TAG@HH:MM:SS` (one block -- necessary as soon as a tag is repeated
    across passes, which is how any table with replicate blocks is taken).
  * NOT excluded by anything else.  No outlier rejection, no trimming, no "drop the first
    block", no smoothing.  A number this tool prints is built from every run it was given
    minus the two categories above, both of which it lists by name in its own output.

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

THE RULE ABOVE REPLACED ONE THAT CONTRADICTED THE PROTOCOL, and the replacement is the reason
some figures in README.md were restated.  The old rule excluded any block whose runs spanned
more than 3 %.  But a 2-run block is within 3 % by construction and a 3-run block is precisely
one whose first two runs were not -- so "exclude spread > 3 %" meant "discard every block for
which the protocol's own remedy was invoked", and the median the protocol tells you to report
was never reported.  On the TiAl box that was most blocks: Task 8's `tial_poly` took 22 runs
across 8 blocks and four were admitted.  It was also not neutral -- all three TiAl figures moved
the flattering way under it.  TWO of Task 6's four published controls moved and are
restated in README.md alongside Task 8's, with the movement and its sign shown: `cantor_h50`
128.4955 -> 127.6610 and `tial_poly` 182.1295 -> 182.2060, because each had a block that ran
to three runs.  (An earlier version of this paragraph claimed all four were unchanged "because
every one of its included blocks has two runs".  Both halves were false, and running the tool
is what shows it -- which is the whole point of the tool.)

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
ROW = re.compile(r"^(\S+)\s+\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})\s.*?natoms=(\d+).*?"
                 r"\bace_us/site=[\d.]+\s*\(n=\d+\s+runs\(exec order\)=\s*"
                 r"([-\d. ]+?),\s*spread=([\d.]+)\)")

# The COMPARATOR half of the row, anchored on `pace_us/site=` in exactly the same way.
#
# WHY IT IS HERE.  Every published ratio has `pair_style pace recursive` as its denominator,
# and until Task 8 this tool summarised only the numerator -- so a table that pooled several
# blocks had a tool-computed ACE median over a HAND-computed pace median.  That is the same
# split the tool was built to close (a hand-typed 182.28 in the paragraph claiming the figures
# came from the artefact), just moved into the denominator.  Both halves now come from here,
# under one pooling rule and one set of exclusions.
#
# A row taken with `pace=none` ends `pace_ms/step=-` and simply does not match; such a block
# contributes to the ACE statistic and to nothing else.  Each series is then excluded on ITS
# OWN spread -- see the loop below for why that is not the same as excluding the block.
PACE = re.compile(r"^\S+\s.*?\bpace_us/site=[\d.]+\s*\(n=\d+\s+runs\(exec order\)=\s*"
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
                    help="a 2-run block whose runs disagree by more than this, and which has "
                         "no third run, is excluded -- the protocol's escalation was not "
                         "applied to it (default 0.03)")
    ap.add_argument("--exclude", action="append", default=[], metavar="TAG[@HH:MM:SS]=REASON",
                    help="exclude one block (TAG@HH:MM:SS) or every block of a tag (TAG); "
                         "a reason is required")
    ap.add_argument("--series", choices=("ace", "pace", "both"), default="ace",
                    help="which pair style to summarise: the exported library (ace, the "
                         "default and the historical behaviour), the ML-PACE comparator "
                         "(pace), or both with the ratio of the two pooled medians (both)")
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
            # The two halves are read from the SAME line, never joined by tag afterwards: a
            # tag appears once per block and the table takes several blocks per tag, so a
            # tag->runs dictionary would silently keep only the last block of each.
            p = PACE.match(line)
            rows.append((m.group(1), m.group(2), int(m.group(3)),
                         [float(x) for x in m.group(4).split() if x != "-"],
                         float(m.group(5)),
                         [float(x) for x in p.group(1).split() if x != "-"] if p else [],
                         float(p.group(2)) if p else 0.0))

    if not rows:
        return die(f"{a.rows_file}: no rows matched the bench_parity.sh row format "
                   f"({len(text.splitlines())} line(s) read). Has the row format changed?")

    # An exclusion names either a TAG (every block carrying it) or ONE BLOCK, as `TAG@HH:MM:SS`.
    #
    # The per-block form had to be added: this table repeats a tag across passes, so a tag-keyed
    # exclusion aimed at one bad block silently takes out all eight of `tial_poly`'s.  Task 6
    # avoided the problem by giving every block a unique tag (`..._fix3r1`, `_fix3r2`), which
    # works but makes the rows file unreadable and splits a tag's blocks across groups.  The
    # timestamp is already in every row and is unique per block.
    tags = {r[0] for r in rows}
    keys = tags | {f"{r[0]}@{r[1]}" for r in rows}
    for tag in excluded:
        if tag not in keys:
            return die(f"exclusion names {tag!r}, which is not in {a.rows_file}. "
                       f"A stale exclusion silently changes a statistic; fix or remove it. "
                       f"(Name one block as TAG@HH:MM:SS, or a whole tag as TAG.)")

    wanted = a.prefixes
    for w in wanted:
        if not any(t.startswith(w) for t in tags):
            return die(f"no row in {a.rows_file} has a tag starting with {w!r}")

    groups = {}
    order = []
    # WHICH BLOCKS COUNT, AND WHY THE RULE CHANGED.
    #
    # The rule used to be "exclude any block whose runs span more than --max-spread".  That
    # CONTRADICTED the protocol it cited.  `bench_parity.sh` takes two runs and, only if they
    # disagree by more than 3 %, a third -- so a 2-run block is within 3 % by construction, and
    # a 3-run block is one whose first two were not.  "Exclude spread > 3 %" therefore meant
    # "discard every block for which the protocol's own remedy was invoked", and the median the
    # protocol says to report was never reported.  On the TiAl box that was most blocks:
    # Task 8's `tial_poly` took 22 runs across 8 blocks and four were admitted.
    #
    # It also was not neutral.  All three TiAl figures moved the flattering way under it.  A
    # selection rule that discards the protocol's own output and whose bias favours the author
    # is not a disclosure item; it is a defect.
    #
    # THE RULE NOW.  Every block counts unless it is explicitly excluded.  A 2-run block
    # contributes both runs, exactly as before.  A 3-run block contributes ONE value, its
    # MEDIAN -- which is what the protocol says that block measured; its three runs are not
    # three independent samples but one measurement plus the remedy for their disagreement.
    # `--max-spread` now flags a block whose FIRST TWO runs disagree by more than it AND which
    # has no third run, i.e. a block the protocol's escalation was not applied to; such a block
    # is excluded, because it is the one case the protocol leaves unresolved.
    #
    # WHAT THIS MOVED.  Nothing in any all-2-run rows file, which is every figure Task 6
    # published -- verified by re-running its four controls, which are unchanged to the last
    # digit.  Task 8's own figures are restated with the movement and its sign shown, in
    # README.md.
    #
    # Each SERIES is judged on its own runs.  Caught on the first real mixed block: Task 8's
    # opening `cantor_poly` block has an ACE spread of 0.08 % and a PACE spread of 8.67 %,
    # because an off-core job perturbed the comparator -- which streams a 193 MB `.yace` and is
    # far more sensitive to system load than the pinned exported library is.  An EXPLICIT
    # exclusion still applies to both series: a reason good enough to drop a block drops all
    # of it.
    def contribution(vals, spread):
        """(values this block contributes, why) -- see the rule above."""
        if len(vals) >= 3:
            v = sorted(vals)
            return [v[len(v) // 2]], f"included (median of {len(vals)} runs, protocol escalation)"
        if len(vals) == 2 and spread > a.max_spread:
            return [], (f"EXCLUDED: 2 runs spanning {spread * 100:.2f} % > "
                        f"{a.max_spread * 100:.0f} % and no third run -- the protocol's "
                        f"escalation was not applied to this block")
        return list(vals), "included"

    for tag, when, nat, vals, spread, pvals, pspread in rows:
        key = next((w for w in wanted if tag.startswith(w)), None) if wanted else tag
        if key is None:
            continue
        if key not in groups:
            groups[key] = {"nat": nat, "runs": [], "pace": [], "blocks": []}
            order.append(key)
        ex = excluded.get(f"{tag}@{when}", excluded.get(tag))
        if ex is not None:
            why = pwhy = f"EXCLUDED: {ex}"
        else:
            add, why = contribution(vals, spread)
            groups[key]["runs"] += add
            padd, pwhy = contribution(pvals, pspread) if pvals else ([], "no comparator run")
            groups[key]["pace"] += padd
        groups[key]["blocks"].append((tag, vals, spread, why, pvals, pspread, pwhy))

    def median(xs):
        v = sorted(xs)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    rc = 0
    for key in order:
        g = groups[key]
        n = len(g["runs"])
        if n == 0:
            print(f"{key}: EVERY block excluded -- no statistic")
            for tag, vals, spread, why, _, _, _ in g["blocks"]:
                print(f"    {tag:<28} {' '.join(f'{x:.3f}' for x in vals)}   {why}")
            rc = 1
            continue
        med = median(g["runs"])
        nin = sum(1 for b in g["blocks"] if b[3].startswith("included"))
        if a.series in ("ace", "both"):
            print(f"{key}: {nin} of {len(g['blocks'])} block(s) included, {n} run(s), "
                  f"pooled median = {med:.4f} ms/step = {med * 1000 / g['nat']:.1f} us/site")
            for tag, vals, spread, why, _, _, _ in g["blocks"]:
                print(f"    {tag:<28} {' '.join(f'{x:.3f}' for x in vals)}"
                      f"   spread={spread * 100:.2f} %   {why}")
        if a.series in ("pace", "both"):
            np_ = len(g["pace"])
            if np_ == 0:
                # Not an error: `pace=none` is a legitimate way to take a row (every row in
                # rows_task6_fix3.txt was taken that way).  Say so rather than print nothing.
                print(f"{key}: pace -- no comparator runs included "
                      f"({'pace=none' if not any(b[4] for b in g['blocks']) else 'every comparator series excluded'})")
                # Nonzero in BOTH modes that asked for a comparator: under `--series both` a
                # group with no usable pace runs has no ratio, and a table row quoted from a
                # run that exited 0 would be a row with a silently missing denominator.
                rc = 1
                continue
            pmed = median(g["pace"])
            pin = sum(1 for b in g["blocks"] if b[6].startswith("included") and b[4])
            nwith = sum(1 for b in g["blocks"] if b[4])
            print(f"{key}: pace recursive, {pin} of {nwith} block(s) included, {np_} run(s), "
                  f"pooled median = {pmed:.4f} ms/step = "
                  f"{pmed * 1000 / g['nat']:.1f} us/site")
            for tag, vals, spread, why, pvals, pspread, pwhy in g["blocks"]:
                if pvals:
                    print(f"    {tag + ' (pace)':<28} "
                          f"{' '.join(f'{x:.3f}' for x in pvals)}"
                          f"   spread={pspread * 100:.2f} %   {pwhy}")
            if a.series == "both":
                # The published ratio.  Both medians are pooled over the same included
                # blocks, so this is the ratio of two numbers this tool printed above -- not
                # a median of the per-block `ratio=` fields, which would weight a 2-run block
                # the same as a 3-run one.
                print(f"    ratio (pooled ace median / pooled pace median) = {med / pmed:.3f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
