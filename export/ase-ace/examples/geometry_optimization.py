#!/usr/bin/env python3
"""
Geometry optimization with ase-ace.

This example demonstrates:
1. Perturbing atomic positions
2. Running BFGS optimization
3. Tracking convergence

Usage:
    python geometry_optimization.py path/to/model.json
"""

import sys
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from ase_ace import ACECalculator


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print("Usage: python geometry_optimization.py path/to/model.json")
        sys.exit(1)

    print("=" * 60)
    print("ACE Potential - Geometry Optimization Example")
    print("=" * 60)

    # Create a supercell and perturb it
    np.random.seed(42)  # Reproducible
    atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1

    print(f"\nStructure: Si diamond 2x2x2, {len(atoms)} atoms")
    print(f"Random perturbation: 0.1 A std")

    with ACECalculator(model_path, num_threads='auto', timeout=120.0) as calc:
        atoms.calc = calc

        # Initial state
        E_init = atoms.get_potential_energy()
        F_init = atoms.get_forces()
        max_f_init = np.abs(F_init).max()

        print(f"\n--- Initial State ---")
        print(f"Energy: {E_init:.6f} eV")
        print(f"Max force: {max_f_init:.4f} eV/A")

        # Run optimization
        print(f"\n--- Running BFGS Optimization ---")
        opt = BFGS(atoms, logfile='-')  # Print to stdout

        fmax = 0.01  # eV/A
        max_steps = 100
        converged = opt.run(fmax=fmax, steps=max_steps)

        # Final state
        E_final = atoms.get_potential_energy()
        F_final = atoms.get_forces()
        max_f_final = np.abs(F_final).max()

        print(f"\n--- Final State ---")
        print(f"Energy: {E_final:.6f} eV")
        print(f"Max force: {max_f_final:.4f} eV/A")
        print(f"Energy change: {E_final - E_init:.6f} eV")
        print(f"Steps: {opt.nsteps}")
        print(f"Converged: {converged or max_f_final < fmax}")

    print("\n" + "=" * 60)
    print("Optimization completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
