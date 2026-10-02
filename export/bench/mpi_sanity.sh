#!/usr/bin/env bash
# mpi_sanity.sh [tag] [ranks] [steps] -- MPI load-balance sanity for pair_style ace.
#
#     export/bench/mpi_sanity.sh tial_poly_b2 4 100
#
# WHAT IT ASKS.  Not "is it fast" and not "is it correct" -- the gates answer correctness
# (`gate_bench_libs.jl` compares 2 ranks against 1 at 1e-13 relative in energy and 1e-12
# absolute in forces) and the protocol rows answer throughput on ONE core.  This asks the one
# thing neither covers: does the work DIVIDE evenly?  LAMMPS' own timing breakdown prints, for
# each section, `%varavg` = (max - min) / avg over the ranks.  A pair style whose per-site cost
# varied with something the domain decomposition does not balance -- a species-dependent cost,
# a neighbour-count-dependent allocation, a workspace shared between ranks -- would show a
# large `%varavg` on `Pair` while every accuracy gate stayed green.
#
# WHAT WAS EXPECTED, AND WHAT WAS FOUND.  The plan expected a few percent, as `pair_style pace`
# gets on the same box, and the earlier finding's 22-34 % was attributed to host load.  **That
# attribution does not survive being tested.**  On a quiet host, 4 ranks, the 2000-atom TiAl box:
# `pair_style ace` gives Pair %varavg 12-31 % over repeated runs, `pace recursive` 4-7 % on the
# identical box and decomposition.  The WORK is balanced -- Nlocal 496-507 and FullNghs
# 55618-56835, both ~1.1 % -- so it is not the decomposition.  See export/bench/README.md.
#
# `MPI_BIND` passes extra options to mpirun (e.g. MPI_BIND="--bind-to core --map-by core",
# which removes roughly a third of the effect).  Both binaries are launched with the same ones.
#
# WHY IT IS NOT PINNED.  Every other measurement here runs under `taskset -c $CORE`; this one
# must not, because pinning four ranks to one core is exactly the way to manufacture the
# imbalance being looked for.  Note that the exported library's ranks are NOT single-threaded
# even with OMP_NUM_THREADS=1: `ps -T` on a live run shows 1-5 threads per rank (the embedded
# Julia runtime's own), placed by mpirun's default policy onto cores it is also giving to other
# ranks.  It therefore needs a quiet host rather than a quiet core, and it
# records `/proc/loadavg` for the same reason the rows do.
set -uo pipefail

TAG=${1:-tial_poly_b2}
RANKS=${2:-4}
STEPS=${3:-100}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LIBDIR=$REPO/bench_parity
LIB=$LIBDIR/libace_$TAG.so
[ -f "$LIB" ] || { echo "mpi_sanity.sh: no library $LIB" >&2; exit 2; }

case "$TAG" in
  cantor*) BOX=$HERE/box_cantor.lmp; PACEFILE=${CANTOR_PACE:-$HOME/si-ace/spike_yace/cantor/cantor_n10000_exact.yace} ;;
  tial*)   BOX=$HERE/box_tial.lmp;   PACEFILE=${TIAL_PACE:-$LIBDIR/tial_o4_pace.yace} ;;
  *) echo "mpi_sanity.sh: cannot infer a box from tag '$TAG'" >&2; exit 2 ;;
esac

# Same per-library plugin selection as run_task8_table.sh: a B2 library exports
# ace_workspace_new and needs the B2 plugin; anything older needs the pre-workspace one.
if nm -D "$LIB" 2>/dev/null | grep -q ' ace_workspace_new$'; then
  PLUGIN=$REPO/verify_cantor/plugin_build_b2/aceplugin.so
else
  PLUGIN=$REPO/verify_cantor/plugin_build/aceplugin.so
fi
LMP_ACE=${LMP_ACE:-$HOME/lammps/lammps-22Jul2025/build/lmp}
LMP_PACE=${LMP_PACE:-/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-acejl/lmp}
VENV=/storage/eng/essswb/venvs/lammps-jax
# ONE mpirun PER BINARY, derived from that binary's own `libmpi` link.
#
# The two LAMMPS builds here link DIFFERENT MPIs (the plugin build OpenMPI 5.0.3, the ML-PACE
# comparator 4.1.6), so there is no single correct `mpirun` for this script.  A foreign one
# launches N independent serial jobs, each reporting "on 1 procs", and every number below would
# then describe a run that never decomposed -- which is why the rank count is asserted from
# LAMMPS' own output rather than from what was requested.
mpirun_for() {   # mpirun_for <lmp binary>
  local d
  d=$(ldd "$1" 2>/dev/null | awk '/libmpi\.so/{print $3}' | head -1)
  [ -n "$d" ] && d=$(dirname "$(dirname "$d")")/bin/mpirun
  if [ -x "${d:-}" ]; then echo "$d"
  else echo /software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/bin/mpirun; fi
}
MPIRUN=${ACE_MPIRUN:-$(mpirun_for "$LMP_ACE")}
[ -x "$MPIRUN" ] || { echo "mpi_sanity.sh: no usable mpirun ($MPIRUN)" >&2; exit 2; }

export OMP_NUM_THREADS=1
# EVERY RUN KEEPS ITS OWN LOG.  The first version reused two fixed names, so a second
# invocation overwrote the first and the two runs this script is meant to CONTRAST -- one with
# mpirun's defaults, one with explicit binding -- were identical in every retained field.  A
# stamp per invocation makes the retained evidence match the reported claim.
STAMP=$(date +%Y%m%dT%H%M%S)

echo "mpi_sanity.sh  $(date '+%Y-%m-%d %H:%M:%S')  tag=$TAG ranks=$RANKS steps=$STEPS"
echo "  loadavg $(cat /proc/loadavg)"
echo "  MPI_BIND '${MPI_BIND:-<mpirun defaults>}'"
echo "  lib     $LIB"
echo "  sha256  $(sha256sum "$LIB" | cut -c1-16)"
echo "  plugin  $PLUGIN"
echo "  mpirun  $MPIRUN"
echo "  box     $(basename "$BOX")"

# `Pair` row of LAMMPS' MPI timing breakdown:
#   Pair    | 1.0349     | 1.0447     | 1.0533     |   0.6 | 99.60
#            min          avg          max          %varavg  %total
report() {   # report <label> <screen log>
  local label=$1 log=$2
  local procs pair loop comm
  # A MISSING OR EMPTY LOG IS A FAILED RUN, NOT A MISSING FIELD.  Found the hard way: a second
  # invocation whose mpirun rejected its options wrote nothing, the previous run's log was
  # still on disk, and the parse below reported that run's figures a second time -- identical
  # to four decimal places, which is the only reason it was noticed.  Every caller deletes the
  # log before the run; this refuses to parse one that is not there.
  if [ ! -s "$log" ]; then
    echo "  $label: NO OUTPUT -- the run failed before LAMMPS wrote anything" >&2
    return 1
  fi
  procs=$(grep -m1 -oE "on [0-9]+ procs" "$log" | awk '{print $2}')
  if [ "${procs:-0}" != "$RANKS" ]; then
    echo "  $label: LAMMPS reports ${procs:-no} rank(s), $RANKS requested -- REFUSING to report" >&2
    echo "    (a foreign mpirun launches independent serial jobs; the figure would be meaningless)" >&2
    return 1
  fi
  loop=$(grep -m1 "Loop time of" "$log" | awk '{print $4}')
  pair=$(awk -F'|' '/^Pair +\|/{gsub(/ /,"",$5); gsub(/ /,"",$6); print $5, $6; exit}' "$log")
  comm=$(awk -F'|' '/^Comm +\|/{gsub(/ /,"",$5); print $5; exit}' "$log")
  echo "  $label: procs=$procs  loop=${loop}s  Pair %varavg=$(echo "$pair" | awk '{print $1}')  %total=$(echo "$pair" | awk '{print $2}')  Comm %varavg=$comm"
  # The balance of the WORK, so that a Pair imbalance can be told apart from a decomposition
  # imbalance.  If Nlocal and FullNghs are even and Pair time is not, the decomposition is fine
  # and the variation is inside the pair style.
  grep -m1 "Nlocal:" "$log" | sed 's/^/    /'
  grep -m1 "FullNghs:" "$log" | sed 's/^/    /'
}

BASE_LD=${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH="$(dirname "$LIB"):/software/easybuild/software/GCCcore/14.3.0/lib64:$HOME/miniconda3/envs/noteable_base_chemistry/lib:/software/easybuild/software/CUDA/12.9.1/lib64:$BASE_LD"
ACE_LOG=$LIBDIR/mpi_sanity_${TAG}_${STAMP}_ace.log
PS_LOG=$LIBDIR/mpi_sanity_${TAG}_${STAMP}_psT.txt
rm -f "$ACE_LOG"
"$MPIRUN" ${MPI_BIND:-} -np "$RANKS" "$LMP_ACE" -in "$HERE/in.bench_ace" -log none -screen "$ACE_LOG" \
    -var box "$BOX" -var steps "$STEPS" -var lib "$LIB" -var plugin "$PLUGIN" >/dev/null 2>&1 &
ACE_PID=$!
# THE THREAD CENSUS, CAPTURED RATHER THAN ASSERTED.  `OMP_NUM_THREADS=1` does not make a rank
# single-threaded: the model library embeds the Julia runtime, which starts threads of its own,
# and mpirun places them by its own policy.  Whether that is happening is the first question to
# ask about a Pair-time imbalance on balanced work, so the evidence is written to a file beside
# the screen log instead of being quoted from a terminal.
{ echo "# ps -T census of the $RANKS ranks, MPI_BIND='${MPI_BIND:-<defaults>}', $(date '+%F %T')"
  echo "# loadavg $(cat /proc/loadavg)"
  sleep 6
  for pid in $(pgrep -P $ACE_PID -f "$(basename "$LMP_ACE")" 2>/dev/null || pgrep -f "$(basename "$LMP_ACE") -in $HERE/in.bench_ace" | head -"$RANKS"); do
    printf '%s\n' "pid $pid: $(ps -T -p "$pid" --no-headers 2>/dev/null | wc -l) thread(s)  cores: $(ps -T -p "$pid" -o psr= 2>/dev/null | tr '\n' ' ')"
  done
  echo "# (a count of 0 means the run had already finished -- take more steps)"
} > "$PS_LOG" 2>&1
wait $ACE_PID
report "pair_style ace " "$ACE_LOG"; RC=$?
echo "  ps -T census:"; sed 's/^/    /' "$PS_LOG"

if [ -f "$PACEFILE" ]; then
  export LD_LIBRARY_PATH="$(dirname "$LMP_PACE"):$VENV/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:$BASE_LD"
  PACE_LOG=$LIBDIR/mpi_sanity_${TAG}_${STAMP}_pace.log
  rm -f "$PACE_LOG"
  PACE_MPIRUN=${ACE_MPIRUN:-$(mpirun_for "$LMP_PACE")}
  echo "  mpirun (pace) $PACE_MPIRUN"
  "$PACE_MPIRUN" ${MPI_BIND:-} -np "$RANKS" "$LMP_PACE" -in "$HERE/in.bench_pace" -log none -screen "$PACE_LOG" \
      -var box "$BOX" -var steps "$STEPS" -var yace "$PACEFILE" >/dev/null 2>&1
  report "pace recursive " "$PACE_LOG" || RC=1
else
  echo "  pace recursive: no comparator file at $PACEFILE -- skipped"
fi

echo "  screen logs: $ACE_LOG ${PACE_LOG:-}"
echo "  ps -T log:   $PS_LOG"
exit $RC
