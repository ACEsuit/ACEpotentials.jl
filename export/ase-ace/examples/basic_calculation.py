#!/usr/bin/env python3
"""
Basic single-point calculation with ase-ace.

This example demonstrates:
1. Loading an ACE potential
2. Computing energy, forces, and stress
3. Using the context manager for cleanup

Usage:
    python basic_calculation.py path/to/model.json
"""

import sys
import numpy as np
from ase.build import bulk
from ase_ace import ACECalculator


def main():
    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print("Usage: python basic_calculation.py path/to/model.json")
        print("\nUsing placeholder - replace with actual model path")
        model_path = "model.json"

    print("=" * 60)
    print("ACE Potential - Basic Calculation Example")
    print("=" * 60)

    # Create a silicon diamond structure
    atoms = bulk('Si', 'diamond', a=5.43)
    print(f"\nStructure: Si diamond, {len(atoms)} atoms")
    print(f"Cell:\n{atoms.cell}")

    # Use ACECalculator with context manager
    print(f"\nLoading model: {model_path}")
    print("Starting Julia driver (first run may take 5-10 seconds)...")

    try:
        with ACECalculator(model_path, num_threads=4, timeout=120.0) as calc:
            atoms.calc = calc

            # Single-point calculation
            print("\n--- Single Point Calculation ---")
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()

            print(f"Energy: {energy:.6f} eV")
            print(f"Energy per atom: {energy/len(atoms):.6f} eV/atom")
            print(f"Max force: {np.abs(forces).max():.6f} eV/A")

            # Pressure from stress (GPa)
            pressure = -stress[:3].mean() * 160.21766208
            print(f"Pressure: {pressure:.2f} GPa")

            # Create a supercell
            print("\n--- Supercell Calculation ---")
            supercell = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
            supercell.calc = calc

            E_super = supercell.get_potential_energy()
            print(f"Structure: Si diamond 2x2x2, {len(supercell)} atoms")
            print(f"Energy: {E_super:.6f} eV")
            print(f"Energy per atom: {E_super/len(supercell):.6f} eV/atom")

            # Verify consistency
            ratio = E_super / energy
            expected_ratio = len(supercell) / len(atoms)
            print(f"\nEnergy ratio (should be ~{expected_ratio}): {ratio:.4f}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please provide a valid path to an ACE model JSON file.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("Make sure Julia is installed and the Julia packages are set up.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Calculation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
