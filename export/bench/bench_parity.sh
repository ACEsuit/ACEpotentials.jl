#!/usr/bin/env bash
# bench_parity.sh <tag> <libace.so> <pace-file|none> [steps]
#
# One row of the parity table, taken under the protocol in export/bench/README.md:
#   * single MPI rank pinned to one core with `taskset -c $CORE`, OMP_NUM_THREADS=1
#   * `timestep 0.0`, $steps (default 100) steps, so every step is one force evaluation
#   * each pair style run TWICE; if the two disagree by more than 3 % a THIRD is taken.
#     The reported value is the SAMPLE MEDIAN in both branches -- for n = 2 that is the
#     midpoint of the two, for n = 3 the middle one.  One statistic, both branches, named
#     in the row as `ace_ms/step(median)`.  The runs are printed in EXECUTION ORDER
#     (`runs(exec order)=...`), never sorted, so any other statistic can be recomputed and
#     so that monotonic drift (throttling) is distinguishable from random scatter.
#   * `pair_style pace recursive` is the comparator, run here, now, on the same core
#   * /proc/loadavg and the date are recorded in the row itself
#
# THE GATE INTERLOCK.  This script does not measure accuracy, but it refuses to time a
# library that has not been gated.  `verify_bench_models.jl` writes `<tag>.gated` next to the
# library and `gate_bench_libs.jl` adds the library-level gates and the library's sha256;
# this script recomputes that sha and refuses on a missing, stale or mismatched manifest.
# `ALLOW_UNGATED="<reason>"` overrides it entirely -- the reason is then printed INTO the row
# as `gate=UNGATED(<reason>)`.  `ALLOW_PARTIAL_GATE=1` is the narrower escape hatch for a
# library that HAS been gated but did not pass every gate: the row then reads
# `gate=PARTIAL[<what failed>]`.  Neither loosens any tolerance; both make the row carry its
# own provenance so an ungated number can never be mistaken for a gated one.
#
# Positional arguments (stable; Tasks 5-8 call this):
#   1 tag        short name for the row, e.g. cantor_poly
#   2 libace.so  the compiled juliac library for `pair_coeff * * <lib> <species>`
#   3 pace-file  the .yace/.ace comparator, or the literal `none` to time only pair_style ace
#   4 steps      optional, default 100
#
# Environment:
#   CORE          core to pin to (default 31)
#   BOX           cantor | tial | <path to a .lmp box file>.  Default: inferred from the tag.
#   PLUGIN        path to aceplugin.so
#   OUT           file to append the row to (default bench_parity/rows.txt)
#   MANIFEST      path to the <tag>.gated manifest (default: beside the library)
#   ALLOW_UNGATED non-empty reason string; times a library with no valid manifest
#   ALLOW_PARTIAL_GATE  times a gated library whose manifest says library_gates=FAIL
#   LMP_ACE / LMP_PACE   override the two LAMMPS binaries
#
# Artefacts kept beside $OUT for every row, successful or not (~4 KB each):
#   <OUT dir>/screen/<tag>_<stamp>_{ace,pace}<n>.log   the full LAMMPS screen log of each run
# They are the only record of atom count, `Ave neighs/atom`, `Neighbor list builds`, the
# thermo energy and the Pair-vs-Loop timing split, none of which appear in the row.
set -uo pipefail

TAG=${1:?usage: bench_parity.sh <tag> <libace.so> <pace-file|none> [steps]}
LIB=$(readlink -f "${2:?missing libace.so}")
PACE_ARG=${3:?missing pace file or 'none'}
STEPS=${4:-100}
CORE=${CORE:-31}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# ---- box -------------------------------------------------------------------------------
BOX=${BOX:-}
if [ -z "$BOX" ]; then
  case "$TAG" in
    cantor*) BOX=cantor ;;
    tial*)   BOX=tial ;;
    *) echo "bench_parity.sh: cannot infer BOX from tag '$TAG'; set BOX=cantor|tial|<file>" >&2
       exit 2 ;;
  esac
fi
case "$BOX" in
  cantor|tial) BOXFILE=$HERE/box_$BOX.lmp ;;
  *)           BOXFILE=$(readlink -f "$BOX") ;;
esac
[ -f "$BOXFILE" ] || { echo "bench_parity.sh: no such box file: $BOXFILE" >&2; exit 2; }

PLUGIN=${PLUGIN:-$REPO/verify_cantor/plugin_build/aceplugin.so}
LMP_ACE=${LMP_ACE:-$HOME/lammps/lammps-22Jul2025/build/lmp}
# The ML-PACE comparator build.  NOTE: this is `-acejl`, not the `-mlpace` build the plan's
# environment notes name.  The -mlpace build's newer ace-evaluator rejects BOTH comparator
# files with `Exception: bad conversion` -- the Cantor .yace written by the wcwitt fork's
# export2lammps_n and the TiAl .yace written by pyace 0.2.8 alike.  The -acejl build (ACE
# version 2023.11.25) reads both and prints "Recursive evaluator is used", so both rows are
# taken with one binary and are comparable.
LMP_PACE=${LMP_PACE:-/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-acejl/lmp}
VENV=/storage/eng/essswb/venvs/lammps-jax
OUTFILE=${OUT:-$REPO/bench_parity/rows.txt}
OUTDIR=$(dirname "$OUTFILE")
STAMP=$(date +%Y%m%dT%H%M%S)
SCREENDIR=$OUTDIR/screen
mkdir -p "$SCREENDIR"

# ---- identity of the library under test (I1) -------------------------------------------
# A basename is NOT an identity: bench_parity/libace_cantor_poly.so (gated at 1.9e-14) and
# verify_cantor/lib/libace_cantor_poly.so (the pre-Task-1 build, failing at 6.88 eV/A) share
# one.  The row carries the resolved path, a sha256 and the mtime.
LIB_SHA=$(sha256sum "$LIB" | cut -c1-16)
LIB_MTIME=$(date -r "$LIB" +%Y-%m-%dT%H:%M:%S)

# ---- gate interlock (I6) ---------------------------------------------------------------
# <tag>.gated is keyed on the library FILE NAME, not on $TAG, so that a row tagged
# `tial_poly_rep3` still resolves the `tial_poly` manifest.
LIBBASE=$(basename "$LIB" .so); LIBTAG=${LIBBASE#libace_}
MANIFEST=${MANIFEST:-$(dirname "$LIB")/${LIBTAG}.gated}
GATE_FIELD="MISSING"
if [ -n "${ALLOW_UNGATED:-}" ]; then
  GATE_FIELD="UNGATED($ALLOW_UNGATED)"
  echo "bench_parity.sh: WARNING -- timing an ungated library on purpose: $ALLOW_UNGATED" >&2
elif [ ! -f "$MANIFEST" ]; then
  echo "bench_parity.sh: REFUSING to time $LIB" >&2
  echo "  no gate manifest at $MANIFEST." >&2
  echo "  Run:  julia --project=export export/bench/verify_bench_models.jl $LIBTAG" >&2
  echo "        export/bench/compile_bench_libs.sh $LIBTAG" >&2
  echo "        julia --project=export export/bench/gate_bench_libs.jl $LIBTAG" >&2
  echo "  or set ALLOW_UNGATED=\"<why>\" to record an explicitly ungated diagnostic row." >&2
  exit 3
else
  WANT_SHA=$(awk -F= '$1=="lib_sha256"{print $2}' "$MANIFEST" | tail -1)
  GATE_SUMMARY=$(awk -F= '$1=="gates"{print $2}' "$MANIFEST" | tail -1)
  if [ -z "$WANT_SHA" ]; then
    echo "bench_parity.sh: REFUSING to time $LIB" >&2
    echo "  $MANIFEST has no lib_sha256 line -- the library-level gates were never run." >&2
    echo "  Run:  julia --project=export export/bench/gate_bench_libs.jl $LIBTAG" >&2
    exit 3
  fi
  FULL_SHA=$(sha256sum "$LIB" | cut -d' ' -f1)
  if [ "$WANT_SHA" != "$FULL_SHA" ]; then
    echo "bench_parity.sh: REFUSING to time $LIB" >&2
    echo "  $MANIFEST gates sha256 $WANT_SHA" >&2
    echo "  the library on disk is    $FULL_SHA" >&2
    echo "  Re-run verify_bench_models.jl / compile_bench_libs.sh / gate_bench_libs.jl." >&2
    exit 3
  fi
  VERDICT=$(awk -F= '$1=="library_gates"{print $2}' "$MANIFEST" | tail -1)
  if [ "$VERDICT" != "PASS" ]; then
    if [ -n "${ALLOW_PARTIAL_GATE:-}" ]; then
      GATE_FIELD="PARTIAL[${GATE_SUMMARY:-unsummarised}]"
      echo "bench_parity.sh: WARNING -- $LIBTAG did not pass every gate ($GATE_SUMMARY);" >&2
      echo "  timing it anyway because ALLOW_PARTIAL_GATE is set.  The row says PARTIAL." >&2
    else
      echo "bench_parity.sh: REFUSING to time $LIB" >&2
      echo "  $MANIFEST says library_gates=$VERDICT ($GATE_SUMMARY)." >&2
      echo "  Fix the gate, or set ALLOW_PARTIAL_GATE=1 to record a row that says PARTIAL." >&2
      exit 3
    fi
  else
    GATE_FIELD="OK[${GATE_SUMMARY:-unsummarised}]"
  fi
fi

export OMP_NUM_THREADS=1
LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg)
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)

# ---- one timed run ---------------------------------------------------------------------
# Echoes the ms/step on stdout, or the empty string if the run failed.  The screen log is
# KEPT in every case (I2): without it nothing records the atom count, `Ave neighs/atom`,
# `Neighbor list builds`, the thermo energy or the Pair-vs-Loop split, so a later change to
# neighbour handling could move a row by 15 % with no artefact revealing why.
run() {   # run <lmp> <input> <label> [extra -var args...]
  local lmp=$1 in=$2 label=$3; shift 3
  local screen="$SCREENDIR/${TAG}_${STAMP}_${label}.log"
  taskset -c "$CORE" "$lmp" -in "$in" -log none -screen "$screen" \
          -var box "$BOXFILE" -var steps "$STEPS" "$@" >/dev/null 2>&1
  local t
  t=$(grep -m1 "Loop time of" "$screen" | awk '{print $4}')
  if [ -z "$t" ]; then
    mv "$screen" "$SCREENDIR/FAILED_${TAG}_${STAMP}_${label}.log" 2>/dev/null
    echo ""
    return 1
  fi
  awk -v t="$t" -v s="$STEPS" 'BEGIN{printf "%.4f", 1000*t/s}'
}

# The SAMPLE MEDIAN of the values given (n = 2 -> midpoint, n = 3 -> middle), the min/max
# spread, and the values in EXECUTION ORDER.  One statistic for both branches; the protocol
# in README.md ("two runs within 3 %, else a third and the median") fixes the sample SIZE,
# this function fixes the ESTIMATOR.
stats() {   # stats <v1> <v2> [v3]  ->  "<median> <n> <v1> <v2> <v3|-> <spread>"
  printf '%s\n' "$*" | awk '{
    n = split($0, x, " ");
    for (i = 1; i <= n; i++) v[i] = x[i];
    for (i = 1; i <= n; i++) for (j = i+1; j <= n; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
    med = (n % 2) ? v[(n+1)/2] : (v[n/2] + v[n/2+1]) / 2.0;
    printf "%.4f %d %s %s %s %.4f", med, n, x[1], x[2], (n >= 3 ? x[3] : "-"), (v[n] - v[1]) / v[1];
  }'
}

series() {   # series <lmp> <input> <label-prefix> [extra args...]
  local lmp=$1 in=$2 pre=$3; shift 3
  local a b c
  a=$(run "$lmp" "$in" "${pre}1" "$@") || { echo "FAILED"; return 1; }
  b=$(run "$lmp" "$in" "${pre}2" "$@") || { echo "FAILED"; return 1; }
  if awk -v a="$a" -v b="$b" 'BEGIN{m=(a<b?a:b); exit !(((a>b?a-b:b-a)/m) > 0.03)}'; then
    c=$(run "$lmp" "$in" "${pre}3" "$@") || { echo "FAILED"; return 1; }
    stats "$a" "$b" "$c"
  else
    stats "$a" "$b"
  fi
}

# ---- pair_style ace --------------------------------------------------------------------
# Same recipe as verify_cantor/run_lammps.sh: the 22Jul2025 build needs GLIBCXX_3.4.32 from
# GCCcore 14.3.0, libpython3.12 from the miniconda env, and libcudart from CUDA 12.9.1.  Without
# all three the binary does not even start ("error while loading shared libraries").
# Saved and restored around the pace block (M4) so the exported library's directory is not on
# the path while the comparator runs.
BASE_LD=${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH="$(dirname "$LIB"):/software/easybuild/software/GCCcore/14.3.0/lib64:$HOME/miniconda3/envs/noteable_base_chemistry/lib:/software/easybuild/software/CUDA/12.9.1/lib64:$BASE_LD"
# IN_ACE overrides the pair_style ace input.  It exists for ONE purpose: `in.bench_ace` sets
# `thermo 50`, so LAMMPS asks for the virial on 2 steps in 100, and since Task 6 the plugin
# calls the cheaper forces-only entry on the other 98.  `in.bench_ace_thermo1` sets
# `thermo 1`, which forces the virial on every step -- the workload every row before Task 6
# measured.  Quoting an IN_ACE row next to a default row is how the virial skip is separated
# from the kernel change; a row taken with a non-default input says so in its `input=` field.
IN_ACE=${IN_ACE:-$HERE/in.bench_ace}
[ -f "$IN_ACE" ] || { echo "bench_parity.sh: no such ace input: $IN_ACE" >&2; exit 2; }
ACE=$(series "$LMP_ACE" "$IN_ACE" ace -var lib "$LIB" -var plugin "$PLUGIN")

# ---- pair_style pace recursive ---------------------------------------------------------
# M5: a failed ACE series short-circuits.  Running the comparator anyway only burns core
# minutes on a shared host for a row that is discarded.
if [ "$ACE" = "FAILED" ]; then
  PACE="- 0 - - - -"; PACEFILE=${PACE_ARG}
  echo "bench_parity.sh: the pair_style ace series FAILED; skipping the pace comparator" >&2
elif [ "$PACE_ARG" = "none" ]; then
  PACE="- 0 - - - -"
  PACEFILE=none
else
  PACEFILE=$(readlink -f "$PACE_ARG")
  # The build directory MUST precede $VENV/lib: BUILD_SHARED_LIBS=ON puts every style in
  # liblammps.so, and with the venv first a stale library is loaded and pair_style pace
  # silently disappears.  (Recipe: ~/si-ace/ACEpotentials/acejax/bench/run_bench.sh)
  export LD_LIBRARY_PATH="$(dirname "$LMP_PACE"):$VENV/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:$BASE_LD"
  PACE=$(series "$LMP_PACE" "$HERE/in.bench_pace" pace -var yace "$PACEFILE")
fi

# ---- atoms per box, for the us/site column (M6) -----------------------------------------
NATOMS=$(grep -m1 -Eo "with [0-9]+ atoms" "$SCREENDIR"/${TAG}_${STAMP}_ace1.log 2>/dev/null | awk '{print $2}')
NATOMS=${NATOMS:-0}

# ---- row -------------------------------------------------------------------------------
ROW=$(awk -v tag="$TAG" -v date="$DATE" -v tm="$TIME" -v core="$CORE" -v la="$LOADAVG" \
          -v steps="$STEPS" -v box="$(basename "$BOXFILE")" -v nat="$NATOMS" \
          -v lib="$LIB" -v libsha="$LIB_SHA" -v libmt="$LIB_MTIME" -v gate="$GATE_FIELD" \
          -v plug="$PLUGIN" -v inace="$(basename "$IN_ACE")" \
          -v pacef="$PACEFILE" -v screen="$SCREENDIR/${TAG}_${STAMP}_*.log" \
          -v ace="$ACE" -v pace="$PACE" '
function us(ms) { return (nat+0 > 0 && ms != "FAILED" && ms != "-") ? sprintf("%.3f", 1000*ms/nat) : "-" }
BEGIN {
  na = split(ace, A, " "); np = split(pace, P, " ");
  am = (na >= 1 ? A[1] : "FAILED"); pm = (np >= 1 ? P[1] : "FAILED");
  ratio = "-";
  if (am != "FAILED" && pm != "FAILED" && pm != "-" && pm+0 > 0) ratio = sprintf("%.2f", am/pm);
  printf "%-18s %s %s core=%-2s loadavg=\"%s\" box=%s natoms=%s steps=%s", tag, date, tm, core, la, box, nat, steps;
  printf " gate=%s lib=%s lib_sha256=%s lib_mtime=%s pace=%s", gate, lib, libsha, libmt, pacef;
  printf " plugin=%s input=%s", plug, inace;
  printf "  ace_ms/step(median)=%s ace_us/site=%s (n=%s runs(exec order)= %s %s %s, spread=%s)", am, us(am), A[2], A[3], A[4], A[5], A[6];
  if (pm == "-") printf "  pace_ms/step=-";
  else printf "  pace_recursive_ms/step(median)=%s pace_us/site=%s (n=%s runs(exec order)= %s %s %s, spread=%s)", pm, us(pm), P[2], P[3], P[4], P[5], P[6];
  printf "  ratio=%s  screen=%s\n", ratio, screen;
}')
echo "$ROW"
echo "$ROW" >> "$OUTFILE"
case "$ROW" in *FAILED*) exit 1 ;; esac
