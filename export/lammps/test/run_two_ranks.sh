#!/bin/bash
# run_two_ranks.sh -- one MPI rank vs two, same geometry, same library, gate 1e-12.
#
# What this proves: `pair_style ace` gets the same forces and the same total energy when
# LAMMPS splits the cell across two domains as when it does not.  A ghost-atom or
# half/full-neighbour-list mistake in the plugin shows up here and essentially nowhere else,
# because a serial run never exercises the ghost path.
#
# Why the geometry comes from a file: `displace_atoms ... random` does NOT produce the same
# perturbation under a different domain decomposition, so building the cell independently in
# the two runs would compare two different geometries.  Step 0 below builds and perturbs the
# cell in a serial run and writes it with `write_data`; both timed runs then `read_data` it.
#
#   run_two_ranks.sh --lmp <lmp> --mpirun <mpirun> --plugin <aceplugin.so> \
#                    --lib <libace.so> [--workdir DIR] [--tol 1e-12] [--size 2] \
#                    [--species "Si"] [--geom <geom.data | box.lmp>]
#
# SPECIES AND GEOMETRY.  Until Task 6 both were hard-coded: a diamond-Si cell and
# `pair_coeff * * <lib> Si`.  That silently restricted the only rank-to-rank gate in the plan
# to single-species models -- run it on the 5-species Cantor library and LAMMPS rejects the
# pair_coeff line, or worse, maps every type to Cr.  `--species` names the pair_coeff element
# list (default "Si"), and `--geom` supplies the cell instead of building one:
#
#   * a `.data` file is `read_data`-ed directly (it must carry the same number of types as
#     `--species` has entries);
#   * anything else is `include`-d as a LAMMPS input fragment that must leave a built box --
#     `export/bench/box_cantor.lmp` and `box_tial.lmp` are exactly such fragments.  Pass
#     `--species` as well: the two timed runs read_data the written cell and never include
#     the fragment, so a `${species}` it defines is not in scope for their pair_coeff line.
#
# The built-in default remains the perturbed diamond-Si cell, so existing call sites are
# unchanged.
#
# Anything not given is taken from the environment (LMP, MPIRUN, ACE_PLUGIN, ACE_LIB) or, for
# `lmp`/`mpirun`, from PATH.  LD_LIBRARY_PATH is used as inherited -- the caller is
# responsible for it (export/test/lammps_harness.jl builds a working one; CI sets it).
#
# Prints, and exits non-zero unless, `TWO_RANK_PARITY PASS`.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$(command -v lmp || true)}"
MPIRUN="${MPIRUN:-$(command -v mpirun || true)}"
PLUGIN="${ACE_PLUGIN:-}"
LIB="${ACE_LIB:-}"
WORKDIR=""
TOL="1e-12"
SIZE=2          # lattice cells per side; 2 -> 64 atoms, enough for a real 2-domain split
SPECIES="Si"
GEOM=""         # empty -> build the default diamond-Si cell (step 0 below)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmp)     LMP="$2"; shift 2 ;;
    --mpirun)  MPIRUN="$2"; shift 2 ;;
    --plugin)  PLUGIN="$2"; shift 2 ;;
    --lib)     LIB="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --tol)     TOL="$2"; shift 2 ;;
    --size)    SIZE="$2"; shift 2 ;;
    --species) SPECIES="$2"; shift 2 ;;
    --geom)    GEOM="$2"; shift 2 ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "$*" >&2; echo "TWO_RANK_PARITY FAIL"; exit 2; }

[[ -n "$LMP"    && -x "$LMP"    ]] || fail "no LAMMPS executable (--lmp): '$LMP'"
[[ -n "$MPIRUN" && -x "$MPIRUN" ]] || fail "no mpirun (--mpirun): '$MPIRUN'"
[[ -n "$PLUGIN" && -f "$PLUGIN" ]] || fail "no plugin (--plugin): '$PLUGIN'"
[[ -n "$LIB"    && -f "$LIB"    ]] || fail "no ACE library (--lib): '$LIB'"

WORKDIR="${WORKDIR:-$(mktemp -d)}"
mkdir -p "$WORKDIR" || fail "cannot create workdir $WORKDIR"
LIB="$(readlink -f "$LIB")"
PLUGIN="$(readlink -f "$PLUGIN")"
[[ -n "$GEOM" ]] && { GEOM="$(readlink -f "$GEOM")"; [[ -f "$GEOM" ]] || fail "no geometry file: $GEOM"; }
cd "$WORKDIR" || fail "cannot cd to $WORKDIR"

# --- step 0: build the geometry once, serially -----------------------------------------
# Whatever the source, the cell ends up in geom.data, written by a SERIAL run: the two timed
# runs both read_data it, so they provably evaluate the same geometry.  (`displace_atoms
# random` does not reproduce under a different domain decomposition, so building the cell
# independently in each run would compare two different configurations.)
if [[ -n "$GEOM" && "$GEOM" == *.data ]]; then
  cp "$GEOM" geom.data || fail "cannot copy $GEOM"
elif [[ -n "$GEOM" ]]; then
  cat > in.build <<EOF
include ${GEOM}
write_data geom.data
EOF
  "$LMP" -in in.build -log log.build -screen none \
    || fail "geometry build from $GEOM failed; see $WORKDIR/log.build"
else
  cat > in.build <<EOF
units metal
atom_style atomic
boundary p p p
lattice diamond 5.43
region box block 0 ${SIZE} 0 ${SIZE} 0 ${SIZE}
create_box 1 box
create_atoms 1 box
mass 1 28.0855
displace_atoms all random 0.05 0.05 0.05 4242
write_data geom.data
EOF
  "$LMP" -in in.build -log log.build -screen none \
    || fail "geometry build failed; see $WORKDIR/log.build"
fi
[[ -f geom.data ]] || fail "no geom.data was produced"

# --- steps 1 and 2: the same input on 1 and on 2 ranks -----------------------------------
cat > in.parity <<EOF
units metal
atom_style atomic
boundary p p p
read_data geom.data
plugin load ${PLUGIN}
pair_style ace
pair_coeff * * ${LIB} ${SPECIES}
variable e equal pe
thermo_style custom step pe
thermo_modify format float %.17g
dump d all custom 1 \${dumpfile} id type x y z fx fy fz
dump_modify d sort id format float %.17g
run 0
print "ACE_ENERGY \$(v_e:%.17g)"
EOF

"$LMP" -in in.parity -var dumpfile dump.np1 -log log.np1 -screen none \
  || fail "1-rank run failed; see $WORKDIR/log.np1"

# --oversubscribe is not accepted by every MPI; retry without it.
if ! "$MPIRUN" -np 2 --oversubscribe "$LMP" -in in.parity -var dumpfile dump.np2 -log log.np2 -screen none 2>/dev/null; then
  "$MPIRUN" -np 2 "$LMP" -in in.parity -var dumpfile dump.np2 -log log.np2 -screen none \
    || fail "2-rank run failed; see $WORKDIR/log.np2"
fi

# A 2-rank job that silently ran as two independent serial jobs (a foreign mpirun) would
# pass the force comparison while proving nothing, so check the rank count LAMMPS reports.
grep -q "2 by 1 by 1\|1 by 2 by 1\|1 by 1 by 2" log.np2 \
  || fail "log.np2 shows no 2-way domain decomposition -- did mpirun really launch 2 ranks?"

E1=$(grep -m1 "^ACE_ENERGY" log.np1 | awk '{print $2}')
E2=$(grep -m1 "^ACE_ENERGY" log.np2 | awk '{print $2}')
[[ -n "$E1" && -n "$E2" ]] || fail "could not read ACE_ENERGY from the logs"

python3 "$SCRIPT_DIR/compare_dump.py" dump.np2 dump.np1 "$TOL" \
        --energy "$E2" "$E1" --label TWO_RANK_PARITY
status=$?
echo "TWO_RANK_PARITY workdir: $WORKDIR"
exit $status
