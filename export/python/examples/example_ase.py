#!/usr/bin/env python3
"""
Example usage of ACE potentials with ASE (Atomic Simulation Environment).

This script demonstrates:
1. Loading a compiled ACE potential
2. Computing energies, forces, and stress
3. Running geometry optimization
4. Running molecular dynamics

Before running, ensure the library path is set:
    source /path/to/deployment/setup_env.sh

Or:
    export LD_LIBRARY_PATH=/path/to/deployment/lib:$LD_LIBRARY_PATH

Requires:
    pip install ase-ace
"""

import os
import sys
from ase_ace import ACELibraryCalculator as ACECalculator
from ase.build import bulk
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import numpy as np


def main():
    # Path to the compiled ACE library
    # Adjust this path to your deployment
    lib_path = os.environ.get('ACE_LIB_PATH', '../lib/libace_silicon.so')

    if not os.path.exists(lib_path):
        print(f"Error: Library not found at {lib_path}")
        print("Set ACE_LIB_PATH environment variable or adjust the path in this script.")
        sys.exit(1)

    print("=" * 60)
    print("ACE Potential Example with ASE")
    print("=" * 60)

    # Load the ACE potential
    print(f"\nLoading ACE potential from: {lib_path}")
    calc = ACECalculator(lib_path)

    # Print model information
    print(f"  Cutoff: {calc.cutoff:.2f} Å")
    print(f"  Species: {calc.species}")

    # Create a silicon diamond structure
    print("\n--- Single Point Calculation ---")
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    print(f"Structure: Si diamond, {len(atoms)} atoms")
    print(f"Energy: {energy:.6f} eV")
    print(f"Energy per atom: {energy/len(atoms):.6f} eV/atom")
    print(f"Max force: {np.abs(forces).max():.6f} eV/Å")
    print(f"Pressure: {-stress[:3].mean() * 160.21766208:.2f} GPa")

    # Create a larger supercell
    print("\n--- Supercell Calculation ---")
    supercell = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
    supercell.calc = calc

    energy_super = supercell.get_potential_energy()
    print(f"Structure: Si diamond 2x2x2, {len(supercell)} atoms")
    print(f"Energy: {energy_super:.6f} eV")
    print(f"Energy per atom: {energy_super/len(supercell):.6f} eV/atom")

    # Geometry optimization example
    print("\n--- Geometry Optimization ---")
    # Slightly perturb the structure
    perturbed = supercell.copy()
    perturbed.positions += np.random.randn(*perturbed.positions.shape) * 0.05
    perturbed.calc = calc

    print(f"Initial energy: {perturbed.get_potential_energy():.6f} eV")
    print(f"Initial max force: {np.abs(perturbed.get_forces()).max():.4f} eV/Å")

    opt = BFGS(perturbed, logfile=None)
    opt.run(fmax=0.01, steps=50)

    print(f"Final energy: {perturbed.get_potential_energy():.6f} eV")
    print(f"Final max force: {np.abs(perturbed.get_forces()).max():.4f} eV/Å")
    print(f"Optimization steps: {opt.nsteps}")

    # Molecular dynamics example
    print("\n--- Molecular Dynamics (NVE) ---")
    md_atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
    md_atoms.calc = calc

    # Initialize velocities at 300 K
    MaxwellBoltzmannDistribution(md_atoms, temperature_K=300)

    # Remove center of mass motion
    momenta = md_atoms.get_momenta()
    momenta -= momenta.mean(axis=0)
    md_atoms.set_momenta(momenta)

    # Run NVE dynamics
    dyn = VelocityVerlet(md_atoms, timestep=1.0 * units.fs)

    energies = []
    def store_energy():
        e_kin = md_atoms.get_kinetic_energy()
        e_pot = md_atoms.get_potential_energy()
        energies.append((e_kin, e_pot, e_kin + e_pot))

    store_energy()  # Initial energy

    print("Running 100 steps of NVE dynamics...")
    for i in range(10):
        dyn.run(10)
        store_energy()

    energies = np.array(energies)
    print(f"Initial total energy: {energies[0, 2]:.6f} eV")
    print(f"Final total energy: {energies[-1, 2]:.6f} eV")
    print(f"Energy drift: {energies[-1, 2] - energies[0, 2]:.6e} eV")
    print(f"Energy std: {energies[:, 2].std():.6e} eV")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
