#!/usr/bin/env bash
# bench_parity.sh <tag> <libace.so> <pace-file|none> [steps]
#
# One row of the parity table, taken under the protocol in export/bench/README.md:
#   * single MPI rank pinned to one core with `taskset -c $CORE`, OMP_NUM_THREADS=1
#   * `timestep 0.0`, $steps (default 100) steps, so every step is one force evaluation
#   * each pair style run TWICE; if the two disagree by more than 3 % a THIRD run is taken
#     and the median reported (the spread is always printed)
#   * `pair_style pace recursive` is the comparator, run here, now, on the same core
#   * /proc/loadavg and the date are recorded in the row itself
#
# This script does NOT verify anything.  A row it prints is meaningless until the library has
# passed its 1e-12 accuracy gate -- run export/bench/verify_bench_models.jl first and quote the
# measured deviation next to the row.
#
# Positional arguments (stable; Tasks 5-8 call this):
#   1 tag        short name for the row, e.g. cantor_poly
#   2 libace.so  the compiled juliac library for `pair_coeff * * <lib> <species>`
#   3 pace-file  the .yace/.ace comparator, or the literal `none` to time only pair_style ace
#   4 steps      optional, default 100
#
# Environment:
#   CORE     core to pin to (default 31)
#   BOX      cantor | tial | <path to a .lmp box file>.  Default: inferred from the tag prefix.
#   PLUGIN   path to aceplugin.so
#   OUT      file to append the row to (default bench_parity/rows.txt); also printed to stdout
#   LMP_ACE / LMP_PACE   override the two LAMMPS binaries
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
LMP_PACE=${LMP_PACE:-/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-mlpace/lmp}
VENV=/storage/eng/essswb/venvs/lammps-jax
OUTFILE=${OUT:-$REPO/bench_parity/rows.txt}
mkdir -p "$(dirname "$OUTFILE")"

export OMP_NUM_THREADS=1
LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg)
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

# ---- one timed run ---------------------------------------------------------------------
# Echoes the ms/step on stdout, or the empty string if the run failed (the screen log is kept).
run() {   # run <lmp> <input> <label> [extra -var args...]
  local lmp=$1 in=$2 label=$3; shift 3
  local screen="$WORK/$label.log"
  taskset -c "$CORE" "$lmp" -in "$in" -log none -screen "$screen" \
          -var box "$BOXFILE" -var steps "$STEPS" "$@" >/dev/null 2>&1
  local t
  t=$(grep -m1 "Loop time of" "$screen" | awk '{print $4}')
  if [ -z "$t" ]; then
    cp "$screen" "$(dirname "$OUTFILE")/FAILED_${TAG}_${label}.log" 2>/dev/null
    echo ""
    return 1
  fi
  awk -v t="$t" -v s="$STEPS" 'BEGIN{printf "%.4f", 1000*t/s}'
}

# median-of-2-or-3 with the 3 % rule
series() {   # series <lmp> <input> <label-prefix> [extra args...]
  local lmp=$1 in=$2 pre=$3; shift 3
  local a b c
  a=$(run "$lmp" "$in" "${pre}1" "$@") || { echo "FAILED"; return 1; }
  b=$(run "$lmp" "$in" "${pre}2" "$@") || { echo "FAILED"; return 1; }
  local spread
  spread=$(awk -v a="$a" -v b="$b" 'BEGIN{m=(a<b?a:b); printf "%.4f", (a>b?a-b:b-a)/m}')
  if awk -v s="$spread" 'BEGIN{exit !(s>0.03)}'; then
    c=$(run "$lmp" "$in" "${pre}3" "$@") || { echo "FAILED"; return 1; }
    # median of three, and report the full spread of the three
    echo "$a $b $c" | awk '{n=split($0,v," "); for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(v[j]<v[i]){t=v[i];v[i]=v[j];v[j]=t}
                            mn=v[1]; mx=v[n]; printf "%.4f 3 %.4f %.4f %.4f %.4f", v[2], v[1], v[2], v[3], (mx-mn)/mn}'
  else
    echo "$a $b" | awk -v s="$spread" '{mn=($1<$2?$1:$2); mx=($1<$2?$2:$1);
                            printf "%.4f 2 %.4f %.4f - %.4f", (mn+mx)/2, $1, $2, s}'
  fi
}

# ---- pair_style ace --------------------------------------------------------------------
export LD_LIBRARY_PATH="$(dirname "$LIB"):/software/easybuild/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
ACE=$(series "$LMP_ACE" "$HERE/in.bench_ace" ace -var lib "$LIB" -var plugin "$PLUGIN")

# ---- pair_style pace recursive ---------------------------------------------------------
if [ "$PACE_ARG" = "none" ]; then
  PACE="- 0 - - - -"
  PACEFILE=none
else
  PACEFILE=$(readlink -f "$PACE_ARG")
  # The build directory MUST precede $VENV/lib: BUILD_SHARED_LIBS=ON puts every style in
  # liblammps.so, and with the venv first a stale library is loaded and pair_style pace
  # silently disappears.  (Recipe: ~/si-ace/ACEpotentials/acejax/bench/run_bench.sh)
  export LD_LIBRARY_PATH="$(dirname "$LMP_PACE"):$VENV/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:${LD_LIBRARY_PATH:-}"
  PACE=$(series "$LMP_PACE" "$HERE/in.bench_pace" pace -var yace "$PACEFILE")
fi

# ---- row -------------------------------------------------------------------------------
ROW=$(awk -v tag="$TAG" -v date="$DATE" -v tm="$TIME" -v core="$CORE" -v la="$LOADAVG" \
          -v steps="$STEPS" -v box="$(basename "$BOXFILE")" \
          -v lib="$(basename "$LIB")" -v pacef="$(basename "$PACEFILE")" \
          -v ace="$ACE" -v pace="$PACE" '
BEGIN {
  na = split(ace, A, " "); np = split(pace, P, " ");
  am = (na >= 1 ? A[1] : "FAILED"); pm = (np >= 1 ? P[1] : "FAILED");
  ratio = "-";
  if (am != "FAILED" && pm != "FAILED" && pm != "-" && pm+0 > 0) ratio = sprintf("%.2f", am/pm);
  printf "%-14s %s %s core=%-2s loadavg=\"%s\" box=%s steps=%s lib=%s pace=%s", tag, date, tm, core, la, box, steps, lib, pacef;
  printf "  ace_ms/step=%s (runs=%s: %s %s %s, spread=%s)", am, A[2], A[3], A[4], A[5], A[6];
  if (pm == "-") printf "  pace_ms/step=- ";
  else printf "  pace_recursive_ms/step=%s (runs=%s: %s %s %s, spread=%s)", pm, P[2], P[3], P[4], P[5], P[6];
  printf "  ratio=%s\n", ratio;
}')
echo "$ROW"
echo "$ROW" >> "$OUTFILE"
case "$ROW" in *FAILED*) exit 1 ;; esac
