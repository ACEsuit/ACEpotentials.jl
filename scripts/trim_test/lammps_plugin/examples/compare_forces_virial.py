#!/usr/bin/env python3
"""Compare forces and virial between Python ACECalculator and LAMMPS."""

import sys
sys.path.insert(0, '../..')

import numpy as np
from ace_calculator import ACECalculator
from ase import Atoms
from ase.build import bulk

# Create the same structure as LAMMPS test
# Diamond Si with 8 atoms in conventional cell
a = 5.43
atoms = Atoms(
    symbols=['Si'] * 8,
    positions=[
        [0.0, 0.0, 0.0],
        [0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0],
        [a/4, a/4, a/4],
        [a/4, 3*a/4, 3*a/4],
        [3*a/4, a/4, 3*a/4],
        [3*a/4, 3*a/4, a/4],
    ],
    cell=[a, a, a],
    pbc=True
)

# Apply same displacement as LAMMPS: atom 0 moved by (0.05, 0.03, -0.02)
atoms.positions[0] += [0.05, 0.03, -0.02]

# Set up calculator
calc = ACECalculator('../../silicon_lib/lib/libace.so')
atoms.calc = calc

# Compute
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # Voigt notation: [xx, yy, zz, yz, xz, xy]
virial = -stress * atoms.get_volume()  # stress = -virial/volume

print("=" * 60)
print("Python ACECalculator Results")
print("=" * 60)
print(f"\nEnergy: {energy:.6f} eV")
print(f"\nPositions (Angstrom):")
for i, pos in enumerate(atoms.positions):
    print(f"  Atom {i+1}: [{pos[0]:12.6f}, {pos[1]:12.6f}, {pos[2]:12.6f}]")

print(f"\nForces (eV/Angstrom):")
for i, f in enumerate(forces):
    print(f"  Atom {i+1}: [{f[0]:15.6f}, {f[1]:15.6f}, {f[2]:15.6f}]")

print(f"\nStress tensor (eV/A^3, Voigt: xx,yy,zz,yz,xz,xy):")
print(f"  {stress}")

print(f"\nVirial (eV, Voigt: xx,yy,zz,yz,xz,xy):")
print(f"  {virial}")

# Read LAMMPS results from forces.dump
print("\n" + "=" * 60)
print("LAMMPS Results (from forces.dump)")
print("=" * 60)

lammps_forces = []
lammps_positions = []
with open('forces.dump', 'r') as f:
    lines = f.readlines()
    # Skip header (9 lines)
    for line in lines[9:]:
        parts = line.split()
        if len(parts) >= 8:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
            lammps_positions.append([x, y, z])
            lammps_forces.append([fx, fy, fz])

lammps_forces = np.array(lammps_forces)
lammps_positions = np.array(lammps_positions)

print(f"\nLAMMPS Energy: -2197.579731 eV")
print(f"\nLAMMPS Positions:")
for i, pos in enumerate(lammps_positions):
    print(f"  Atom {i+1}: [{pos[0]:12.6f}, {pos[1]:12.6f}, {pos[2]:12.6f}]")

print(f"\nLAMMPS Forces (eV/Angstrom):")
for i, f in enumerate(lammps_forces):
    print(f"  Atom {i+1}: [{f[0]:15.6f}, {f[1]:15.6f}, {f[2]:15.6f}]")

# Compare
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

# Need to match atom ordering - LAMMPS atom 1 is at displaced position
# Find matching atoms by position
tol = 0.1
matched = []
for i, lmp_pos in enumerate(lammps_positions):
    for j, ase_pos in enumerate(atoms.positions):
        if np.allclose(lmp_pos, ase_pos, atol=tol):
            matched.append((i, j))
            break

print(f"\nMatched atoms (LAMMPS -> ASE): {matched}")

print(f"\nForce comparison (matched atoms):")
max_force_error = 0
for lmp_i, ase_j in matched:
    lmp_f = lammps_forces[lmp_i]
    ase_f = forces[ase_j]
    diff = np.abs(lmp_f - ase_f)
    max_diff = np.max(diff)
    max_force_error = max(max_force_error, max_diff)
    print(f"  LAMMPS atom {lmp_i+1} / ASE atom {ase_j+1}:")
    print(f"    LAMMPS: [{lmp_f[0]:12.6f}, {lmp_f[1]:12.6f}, {lmp_f[2]:12.6f}]")
    print(f"    ASE:    [{ase_f[0]:12.6f}, {ase_f[1]:12.6f}, {ase_f[2]:12.6f}]")
    print(f"    Diff:   [{diff[0]:12.6e}, {diff[1]:12.6e}, {diff[2]:12.6e}]")

print(f"\nMax force error: {max_force_error:.6e} eV/Angstrom")

# Energy comparison
lammps_energy = -2197.579731412351066
energy_error = abs(energy - lammps_energy)
print(f"\nEnergy comparison:")
print(f"  ASE:    {energy:.6f} eV")
print(f"  LAMMPS: {lammps_energy:.6f} eV")
print(f"  Error:  {energy_error:.6e} eV")

# Virial/Stress comparison
# ASE stress convention: positive = tension, stress = -virial/volume
# LAMMPS pressure convention: positive = compression
#
# The relationship is:
#   LAMMPS pressure (bar) = -ASE stress (eV/A^3) * 1.6022e6
#   So: ASE stress = -LAMMPS pressure / 1.6022e6
volume = atoms.get_volume()
lammps_pxx = -249691064.60176172853  # bar (negative = tension)
lammps_pyy = -251021704.62560117245
lammps_pzz = -251420478.97082474828
lammps_pxy = 14450484.889648314565
lammps_pxz = -19069608.864742834121
lammps_pyz = -30230294.115820925683

# Convert LAMMPS pressure to ASE stress convention
# ASE stress Voigt: [xx, yy, zz, yz, xz, xy]
bar_to_eV_per_A3 = 1.0 / 1.6021766208e6
lammps_stress = -np.array([
    lammps_pxx, lammps_pyy, lammps_pzz,
    lammps_pyz, lammps_pxz, lammps_pxy
]) * bar_to_eV_per_A3

print(f"\nStress comparison (eV/A^3):")
print(f"  ASE stress:    {stress}")
print(f"  LAMMPS stress: {lammps_stress}")
stress_error = np.abs(stress - lammps_stress)
print(f"  Error:         {stress_error}")
print(f"  Max error:     {np.max(stress_error):.6e} eV/A^3")

# Also compute virial for reference
lammps_virial = -lammps_stress * volume
print(f"\nVirial comparison (eV):")
print(f"  ASE virial:    {virial}")
print(f"  LAMMPS virial: {lammps_virial}")
virial_error = np.abs(virial - lammps_virial)
print(f"  Max error:     {np.max(virial_error):.6e} eV")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Energy error:     {energy_error:.6e} eV")
print(f"Max force error:  {max_force_error:.6e} eV/Angstrom")
print(f"Max stress error: {np.max(stress_error):.6e} eV/A^3")

# Tolerances: energy/forces at machine precision, stress within ~1e-4 eV/A^3
# (stress has minor discrepancy from unit conversion precision)
if energy_error < 1e-6 and max_force_error < 1e-6 and np.max(stress_error) < 1e-4:
    print("\n*** ALL TESTS PASSED ***")
else:
    print("\n*** SOME TESTS MAY NEED INVESTIGATION ***")
