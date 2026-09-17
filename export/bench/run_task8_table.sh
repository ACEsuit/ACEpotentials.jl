#!/usr/bin/env bash
# run_task8_table.sh [outfile] -- the plan's CLOSE-OUT table, taken in ONE session.
#
# Every row of the final parity table in export/bench/README.md comes from this script.  It
# exists so that the table can be re-taken by one command rather than reconstructed from a
# shell history: Task 6 fix round 4 found a README figure that had been typed by hand, and
# Task 7 found two published tables with no artefact at all.  The rule that came out of both
# is that a quoted number must be reproducible from a committed script plus a committed rows
# file.  This is the script; bench_parity/rows_task8.txt (copied to export/bench/artefacts/)
# is the rows file; export/bench/summarise_rows.py is what turns one into the other.
#
# WHAT IT MEASURES.  Both reference models in :polynomial at every step of the plan
# (baseline = Task 4, B1 = Task 5, B2 = Task 6 = shipped), plus one :hermite_spline row per
# model at the shipped generator, each against `pair_style pace recursive` run here, now, on
# the same core -- the comparator is re-run inside every block, so no row is compared against
# a pace number taken on another day.
#
# WHAT IT DOES NOT MEASURE.  B3 (aa_products=:dag) is not in the table.  It is off by default
# and its libraries are gone from disk; its numbers stay in Task 7's section of the README.
#
# PROTOCOL (export/bench/README.md).  Single rank, `taskset -c $CORE`, OMP_NUM_THREADS=1,
# timestep 0.0, 100 steps, two runs per block and a third if they disagree by more than 3 %.
# Blocks are taken in PASSES rather than one tag at a time, so the repeats of a tag are
# separated by ~10 minutes of other work: a session-long drift then shows up as disagreement
# between a tag's own blocks instead of hiding inside a single contiguous block.  That is the
# operational form of the standing rule "one block that agrees with itself is not evidence".
#
# PASS STRUCTURE.  Passes A and B cover all eight tags.  Pass C repeats the three TiAl
# :polynomial tags, because tial_poly carries the +-7 % block-to-block scatter Task 4
# documented and the protocol requires >= 5 runs there; three blocks give six.
#
# Usage:  export/bench/run_task8_table.sh [outfile]
#         CORE=31 (default) -- verify the core is idle with `mpstat -P <core> 1 3` first.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=${1:-$REPO/bench_parity/rows_task8.txt}
CORE=${CORE:-31}

LIBDIR=$REPO/bench_parity
CANTOR_PACE=${CANTOR_PACE:-$HOME/si-ace/spike_yace/cantor/cantor_n10000_exact.yace}
TIAL_PACE=${TIAL_PACE:-$LIBDIR/tial_o4_pace.yace}

for f in "$CANTOR_PACE" "$TIAL_PACE"; do
  [ -f "$f" ] || { echo "run_task8_table.sh: missing pace comparator $f" >&2; exit 2; }
done

# THE PLUGIN IS CHOSEN PER LIBRARY, FROM THE LIBRARY'S OWN SYMBOL TABLE.
#
# Task 6 replaced the C ABI: a B2 library exports `ace_workspace_new` and the B2 plugin calls
# it, while the baseline and B1 libraries export the old stateless entry points.  The two do
# not mix, and both mismatches are loud rather than silent -- a B2 plugin against a legacy
# library stops with "Cannot find ace_workspace_new in model", and the legacy plugin against a
# B2 library stops with LAMMPS' "Non-numeric variable value" (Task 6 recorded that).  This was
# NOT anticipated when the table was planned: the first attempt ran every row against
# bench_parity.sh's default (B2) plugin and the two pre-B2 Cantor blocks failed outright.
#
# Selecting on `nm -D | grep ace_workspace_new` rather than on the tag suffix means the choice
# is made from the binary under test, so a future tag naming scheme cannot silently pair a
# library with the wrong plugin.  It also reproduces the conditions of the rows it is being
# compared against: Tasks 4 and 5 took the baseline and B1 rows with exactly this plugin
# (`plugin=.../plugin_build/aceplugin.so` in rows_task4.txt and rows_task5.txt).  Every row
# records its `plugin=` explicitly, so the pairing travels with the number.
plugin_for() {   # plugin_for <lib.so>
  if nm -D "$1" 2>/dev/null | grep -q ' ace_workspace_new$'; then
    echo "$REPO/verify_cantor/plugin_build_b2/aceplugin.so"
  else
    echo "$REPO/verify_cantor/plugin_build/aceplugin.so"
  fi
}

# tag -> comparator.  The tag is also the library basename (libace_<tag>.so) and the gate
# manifest name (<tag>.gated); bench_parity.sh refuses any tag whose manifest is missing,
# stale or not PASS, so an ungated library cannot enter this table by accident.
run_block() {   # run_block <tag>
  local tag=$1 pace lib plug
  case "$tag" in
    cantor*) pace=$CANTOR_PACE ;;
    tial*)   pace=$TIAL_PACE ;;
    *) echo "run_task8_table.sh: no comparator for tag $tag" >&2; return 2 ;;
  esac
  lib=$LIBDIR/libace_$tag.so
  [ -f "$lib" ] || { echo "run_task8_table.sh: no library $lib" >&2; return 2; }
  plug=$(plugin_for "$lib")
  [ -f "$plug" ] || { echo "run_task8_table.sh: no plugin $plug" >&2; return 2; }
  echo "=== $(date +%H:%M:%S)  $tag  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)  plugin=$(basename "$(dirname "$plug")")"
  CORE=$CORE OUT=$OUT PLUGIN=$plug "$HERE/bench_parity.sh" "$tag" "$lib" "$pace" 100
}

PASS_ALL="cantor_poly cantor_poly_b1 cantor_poly_b2 cantor_h50_b2 \
          tial_poly tial_poly_b1 tial_poly_b2 tial_h50_b2"
PASS_TIAL="tial_poly tial_poly_b1 tial_poly_b2"

echo "### run_task8_table.sh  start $(date '+%Y-%m-%d %H:%M:%S')  core=$CORE"
echo "### loadavg at dispatch: $(cat /proc/loadavg)"
echo "### out: $OUT"

# TAGS overrides the pass structure entirely and runs exactly the tags given, once each, in
# order.  It exists for TOP-UP passes: the 3 % spread rule is the protocol's, and on the TiAl
# box -- which Task 4 measured as carrying +-7 % irreducible block-to-block scatter -- it
# excludes blocks often enough that a tag can finish a full session with fewer included runs
# than the protocol requires.  Topping such a tag up with further blocks is the protocol
# working as intended; it is NOT the same as re-running a tag until a number is liked, and the
# difference is that the top-up blocks go into the SAME rows file and are pooled with the rest
# by `summarise_rows.py`, which shows every block it included and every block it dropped.
#     TAGS="tial_poly tial_poly tial_poly_b2" export/bench/run_task8_table.sh rows.txt
if [ -n "${TAGS:-}" ]; then
  echo "### TOP-UP PASS: $TAGS  $(date +%H:%M:%S)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  for tag in $TAGS; do run_block "$tag"; done
else
  for pass in A B; do
    echo "### PASS $pass  $(date +%H:%M:%S)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
    for tag in $PASS_ALL; do run_block "$tag"; done
  done
  echo "### PASS C (TiAl :polynomial repeats, for n >= 5)  $(date +%H:%M:%S)"
  for tag in $PASS_TIAL; do run_block "$tag"; done
fi

echo "### run_task8_table.sh  done $(date '+%Y-%m-%d %H:%M:%S')  loadavg=$(cat /proc/loadavg)"
echo "### summarise with:"
echo "###   export/bench/summarise_rows.py $OUT --series both"
