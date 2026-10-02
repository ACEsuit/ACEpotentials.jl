#!/bin/bash
# test_omp_vs_serial.sh -- OMP_NUM_THREADS=4 vs 1, same geometry, same library, gate 1e-12.
#
# What this proves, and what it does not.
#
# The exported library keeps all its scratch in a caller-supplied Workspace (Task 6 / B2) and
# the plugin allocates one per OpenMP thread.  If that were wrong -- a workspace shared
# between threads, or `workspaces[tid]` indexed with something that is not stable for the
# parallel region -- two threads would interleave their writes to `A`, `∂A` and the neighbour
# cache and the forces would differ, usually on a handful of atoms and not reproducibly.  So
# the comparison is against the SERIAL run of the same binary, on the same geometry.
#
# THE NON-TRIVIALITY GATE IS THE POINT OF THE SCRIPT.  A plugin built with `-DBUILD_OMP=OFF`
# ignores OMP_NUM_THREADS entirely: both runs are then serial, agree to the last bit, and the
# test passes while proving nothing at all.  This script therefore REQUIRES the 4-thread run
# to report four ACE workspaces -- the plugin prints "ACE: <n> evaluation workspace(s)" from
# init_style, with n = omp_get_max_threads() -- and fails if it does not.  Do not remove that
# check to make the script pass with a serial plugin; rebuild the plugin with
# `-DBUILD_OMP=ON`.
#
#   test_omp_vs_serial.sh --lmp <lmp> --plugin <aceplugin.so> --lib <libace.so> \
#                         [--box <box.lmp>] [--species "Cr Mn Fe Co Ni"] \
#                         [--threads 4] [--tol 1e-12] [--workdir DIR]
#
# The default box is export/bench/box_cantor.lmp at `-var cells 4` (128 atoms, the same
# lattice and species fractions as the timing box), whose forces are O(1) eV/A, so
# compare_dump.py's --min-force gate is meaningful.  `--species` is only needed when the box
# file does not define ${species} itself.
#
# LD_LIBRARY_PATH is used as inherited; the caller is responsible for it.
#
# Prints, and exits non-zero unless, `OMP_PARITY PASS`.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LMP="${LMP:-$(command -v lmp || true)}"
PLUGIN="${ACE_PLUGIN:-}"
LIB="${ACE_LIB:-}"
BOX="$REPO/export/bench/box_cantor.lmp"
SPECIES=""
CELLS=4
THREADS=4
TOL="1e-12"
WORKDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmp)     LMP="$2"; shift 2 ;;
    --plugin)  PLUGIN="$2"; shift 2 ;;
    --lib)     LIB="$2"; shift 2 ;;
    --box)     BOX="$2"; shift 2 ;;
    --species) SPECIES="$2"; shift 2 ;;
    --cells)   CELLS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --tol)     TOL="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "$*" >&2; echo "OMP_PARITY FAIL"; exit 2; }

[[ -n "$LMP"    && -x "$LMP"    ]] || fail "no LAMMPS executable (--lmp): '$LMP'"
[[ -n "$PLUGIN" && -f "$PLUGIN" ]] || fail "no plugin (--plugin): '$PLUGIN'"
[[ -n "$LIB"    && -f "$LIB"    ]] || fail "no ACE library (--lib): '$LIB'"
[[ -f "$BOX" ]] || fail "no box file (--box): '$BOX'"

WORKDIR="${WORKDIR:-$(mktemp -d)}"
mkdir -p "$WORKDIR" || fail "cannot create workdir $WORKDIR"
LIB="$(readlink -f "$LIB")"
PLUGIN="$(readlink -f "$PLUGIN")"
BOX="$(readlink -f "$BOX")"
cd "$WORKDIR" || fail "cannot cd to $WORKDIR"

SPECIES_LINE=""
[[ -n "$SPECIES" ]] && SPECIES_LINE="variable species string \"$SPECIES\""

cat > in.omp <<EOF
include ${BOX}
${SPECIES_LINE}
plugin load ${PLUGIN}
pair_style ace
pair_coeff * * ${LIB} \${species}
variable e equal pe
thermo_style custom step pe press
thermo_modify format float %.17g
dump d all custom 1 \${dumpfile} id type x y z fx fy fz
dump_modify d sort id format float %.17g
run 0
print "ACE_ENERGY \$(v_e:%.17g)"
EOF

for t in 1 "$THREADS"; do
  OMP_NUM_THREADS=$t "$LMP" -in in.omp -var cells "$CELLS" -var dumpfile "dump.omp$t" \
      -log "log.omp$t" -screen none \
    || fail "OMP_NUM_THREADS=$t run failed; see $WORKDIR/log.omp$t"
done

# --- the non-triviality gate: OpenMP must actually be on ---------------------------------
nws=$(grep -m1 -oP 'ACE: \K[0-9]+(?= evaluation workspace)' "log.omp$THREADS" || true)
[[ -n "$nws" ]] || fail "log.omp$THREADS does not report the ACE workspace count -- \
is this plugin older than the workspace API?"
if [[ "$nws" -lt 2 ]]; then
  fail "the OMP_NUM_THREADS=$THREADS run allocated $nws ACE workspace(s), i.e. \
omp_get_max_threads() == $nws: this plugin was built WITHOUT OpenMP (-DBUILD_OMP=OFF), so \
both runs were serial and the comparison proves nothing.  Rebuild with -DBUILD_OMP=ON."
fi
echo "OMP_PARITY: OMP_NUM_THREADS=$THREADS run used $nws ACE workspaces (one per OpenMP thread)"

E1=$(grep -m1 "^ACE_ENERGY" log.omp1 | awk '{print $2}')
E2=$(grep -m1 "^ACE_ENERGY" "log.omp$THREADS" | awk '{print $2}')
[[ -n "$E1" && -n "$E2" ]] || fail "could not read ACE_ENERGY from the logs"

python3 "$SCRIPT_DIR/compare_dump.py" "dump.omp$THREADS" dump.omp1 "$TOL" \
        --energy "$E2" "$E1" --label OMP_PARITY
status=$?
echo "OMP_PARITY workdir: $WORKDIR"
exit $status
