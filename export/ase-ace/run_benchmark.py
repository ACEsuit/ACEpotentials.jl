#!/usr/bin/env python3
"""Benchmark Library Calculator vs LAMMPS."""
import time
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from ase.build import bulk
from ase.io import write as ase_write
from ase.data import atomic_masses
from ase_ace import ACELibraryCalculator

BASE = Path(__file__).parent.parent.resolve()
LIBRARY = str(BASE / "export/test/build/libace_test.so")
PLUGIN = str(BASE / "export/lammps/plugin/build/aceplugin.so")
LAMMPS = "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"

print("=" * 70)
print("ACE Interface Benchmark: Library Calculator vs LAMMPS")
print("=" * 70)

sizes = [2, 3, 4, 5, 7]
results = {}

for n in sizes:
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)
    np.random.seed(42)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
    natoms = len(atoms)

    # Library Calculator
    calc = ACELibraryCalculator(LIBRARY)
    atoms.calc = calc

    atoms.get_potential_energy()
    atoms.get_forces()

    n_iter = 5
    times = []
    for i in range(n_iter):
        atoms.positions[0, 0] += 0.0001
        t0 = time.perf_counter()
        atoms.get_potential_energy()
        atoms.get_forces()
        times.append(time.perf_counter() - t0)

    lib_time = np.mean(times) * 1000

    # LAMMPS
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_file = tmpdir / "test.data"
        ase_write(str(data_file), atoms, format="lammps-data")

        n_steps = 10
        input_script = f"""
units metal
atom_style atomic
boundary p p p
read_data {data_file}
mass 1 {atomic_masses[14]}

plugin load {PLUGIN}
pair_style ace
pair_coeff * * {LIBRARY} Si

fix 1 all nve
timestep 0.0

run 5
reset_timestep 0
run {n_steps}
"""
        input_file = tmpdir / "in.test"
        with open(input_file, "w") as f:
            f.write(input_script)

        def run_lammps(threads):
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)
            # Ensure GCC libraries are available
            gcc_lib = "/software/easybuild/software/GCCcore/14.3.0/lib64"
            if "LD_LIBRARY_PATH" in env:
                env["LD_LIBRARY_PATH"] = f"{gcc_lib}:{env['LD_LIBRARY_PATH']}"
            else:
                env["LD_LIBRARY_PATH"] = gcc_lib
            result = subprocess.run(
                [LAMMPS, "-in", str(input_file)],
                capture_output=True, text=True, env=env, cwd=tmpdir
            )
            if result.returncode != 0:
                if natoms <= 64:  # Print error for small sizes
                    print(f"  LAMMPS returncode: {result.returncode}")
                    print(f"  LAMMPS stderr: {result.stderr}")
                    print(f"  LAMMPS stdout (last 1000): {result.stdout[-1000:]}")
                return None
            for line in result.stdout.split("\n"):
                if "Loop time of" in line:
                    parts = line.split()
                    loop_time = float(parts[3])
                    steps = int(parts[parts.index("for") + 1])
                    if steps == n_steps:
                        return (loop_time / steps) * 1000
            return None

        lammps_1t = run_lammps(1)
        lammps_8t = run_lammps(8)

    results[natoms] = {"library": lib_time, "lammps_1t": lammps_1t, "lammps_8t": lammps_8t}

    print(f"\nN={natoms:4d} atoms:")
    print(f"  Library (1T):  {lib_time:7.2f} ms")
    if lammps_1t:
        print(f"  LAMMPS  (1T):  {lammps_1t:7.2f} ms  ({lib_time/lammps_1t:.1f}x faster)")
    if lammps_8t:
        print(f"  LAMMPS  (8T):  {lammps_8t:7.2f} ms  ({lib_time/lammps_8t:.1f}x faster)")

print("\n" + "=" * 70)
print("Summary Table (ms per E+F evaluation)")
print("=" * 70)
print(f"{'Atoms':>6} {'Library':>10} {'LAMMPS(1T)':>12} {'LAMMPS(8T)':>12} {'Speedup':>10}")
print("-" * 52)
for n in sorted(results.keys()):
    r = results[n]
    speedup = r["library"] / r["lammps_8t"] if r["lammps_8t"] else 0
    l1t = f"{r['lammps_1t']:.2f}" if r['lammps_1t'] else "N/A"
    l8t = f"{r['lammps_8t']:.2f}" if r['lammps_8t'] else "N/A"
    print(f"{n:6d} {r['library']:10.2f} {l1t:>12} {l8t:>12} {speedup:10.1f}x")
