#!/usr/bin/env python3
"""Evaluate a compiled ACE library on a LAMMPS data file and print the result exactly.

    ACE_LIB_PATH=<libace.so> ACE_GEOM=<geom.data> ACE_TYPE_MAP=1:14,2:26 \
        python3 eval_library.py

Line 1 is the total energy, lines 2..N+1 are the force components, one atom per line in
ascending LAMMPS atom-id order.  Every number is printed with `repr`, i.e. with enough
digits to round-trip the double exactly -- the point of this script is to let the Julia side
compare the library against its own evaluation at 1e-12, which a `%.6f` would make
impossible.

`ACE_TYPE_MAP` is optional and defaults to `1:14` (a single Si type).
"""

import os
import sys

import numpy as np
from ase.io import read

from ase_ace import ACELibraryCalculator


def main() -> int:
    lib = os.environ["ACE_LIB_PATH"]
    geom = os.environ["ACE_GEOM"]
    type_map = {
        int(k): int(v)
        for k, v in (p.split(":") for p in os.environ.get("ACE_TYPE_MAP", "1:14").split(","))
    }

    atoms = read(geom, format="lammps-data", style="atomic", Z_of_type=type_map)
    atoms.calc = ACELibraryCalculator(lib)

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)

    print(repr(energy))
    for f in forces:
        print(repr(float(f[0])), repr(float(f[1])), repr(float(f[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
