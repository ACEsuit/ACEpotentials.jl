#!/usr/bin/env python3
"""
Molecular dynamics with ase-ace.

This example demonstrates:
1. Initializing velocities
2. Running NVE dynamics
3. Monitoring energy conservation

Usage:
    python molecular_dynamics.py path/to/model.json
"""

import sys
import numpy as np
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase_ace import ACECalculator


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print("Usage: python molecular_dynamics.py path/to/model.json")
        sys.exit(1)

    print("=" * 60)
    print("ACE Potential - Molecular Dynamics Example")
    print("=" * 60)

    # Create a larger system for MD
    atoms = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)
    print(f"\nStructure: Si diamond 3x3x3, {len(atoms)} atoms")

    with ACECalculator(model_path, num_threads=8, timeout=120.0) as calc:
        atoms.calc = calc

        # Initialize velocities at 300 K
        temperature = 300  # Kelvin
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

        # Remove center of mass motion
        momenta = atoms.get_momenta()
        momenta -= momenta.mean(axis=0)
        atoms.set_momenta(momenta)

        print(f"Initial temperature: {temperature} K")

        # Set up NVE dynamics
        timestep = 1.0  # fs
        dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)

        # Track energies
        energies = []
        temperatures = []

        def store_data():
            e_kin = atoms.get_kinetic_energy()
            e_pot = atoms.get_potential_energy()
            temp = e_kin / (1.5 * len(atoms) * units.kB)
            energies.append((e_kin, e_pot, e_kin + e_pot))
            temperatures.append(temp)

        store_data()  # Initial

        # Run dynamics
        n_steps = 100
        print_interval = 20
        print(f"\n--- Running {n_steps} steps of NVE dynamics ---")
        print(f"Timestep: {timestep} fs")
        print(f"\n{'Step':>6} {'E_kin':>12} {'E_pot':>12} {'E_tot':>12} {'T (K)':>10}")
        print("-" * 56)

        print(f"{0:>6} {energies[0][0]:>12.4f} {energies[0][1]:>12.4f} {energies[0][2]:>12.4f} {temperatures[0]:>10.1f}")

        for i in range(n_steps // print_interval):
            dyn.run(print_interval)
            store_data()
            step = (i + 1) * print_interval
            e = energies[-1]
            t = temperatures[-1]
            print(f"{step:>6} {e[0]:>12.4f} {e[1]:>12.4f} {e[2]:>12.4f} {t:>10.1f}")

        # Summary statistics
        energies = np.array(energies)
        temperatures = np.array(temperatures)

        print(f"\n--- Summary ---")
        print(f"Initial total energy: {energies[0, 2]:.6f} eV")
        print(f"Final total energy: {energies[-1, 2]:.6f} eV")
        print(f"Energy drift: {energies[-1, 2] - energies[0, 2]:.6e} eV")
        print(f"Energy fluctuation (std): {energies[:, 2].std():.6e} eV")
        print(f"Temperature range: {temperatures.min():.1f} - {temperatures.max():.1f} K")

    print("\n" + "=" * 60)
    print("MD simulation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
