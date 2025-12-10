"""
OpenMP-accelerated ACE Potential Calculator

This module provides a multi-threaded version of the ACE calculator
using an OpenMP C wrapper around the Julia-compiled library.

Usage:
    from ace_calculator_omp import ACECalculatorOMP

    calc = ACECalculatorOMP("libace.so", num_threads=4)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional

# Try to import ASE
try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Calculator = object
    all_changes = []


class ACECalculatorOMP(Calculator if HAS_ASE else object):
    """
    OpenMP-accelerated ASE Calculator for ACE potentials.

    This calculator uses an OpenMP C wrapper to parallelize the
    atom loop when computing energies and forces.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
        self,
        ace_library_path: str,
        omp_wrapper_path: Optional[str] = None,
        num_threads: int = 4,
        **kwargs
    ):
        """
        Initialize the OpenMP-accelerated ACE calculator.

        Args:
            ace_library_path: Path to the Julia-compiled ACE .so file
            omp_wrapper_path: Path to libace_omp.so (auto-detected if None)
            num_threads: Number of OpenMP threads to use
        """
        if HAS_ASE:
            Calculator.__init__(self, **kwargs)

        self.ace_lib_path = Path(ace_library_path).resolve()
        if not self.ace_lib_path.exists():
            raise FileNotFoundError(f"ACE library not found: {self.ace_lib_path}")

        # Find OMP wrapper
        if omp_wrapper_path is None:
            # Look in same directory as this file
            module_dir = Path(__file__).parent
            omp_wrapper_path = module_dir / "libace_omp.so"

        self.omp_wrapper_path = Path(omp_wrapper_path).resolve()
        if not self.omp_wrapper_path.exists():
            raise FileNotFoundError(
                f"OpenMP wrapper not found: {self.omp_wrapper_path}\n"
                "Build it with: gcc -shared -fPIC -O3 -fopenmp -o libace_omp.so ace_omp_wrapper.c -ldl -lm"
            )

        # Load wrapper with RTLD_NOW to resolve all symbols immediately
        self.lib = ctypes.CDLL(str(self.omp_wrapper_path), mode=ctypes.RTLD_NOW)
        self._setup_functions()

        # Initialize
        ret = self.lib.ace_omp_init(str(self.ace_lib_path).encode())
        if ret != 0:
            raise RuntimeError(f"Failed to initialize ACE library: {self.ace_lib_path}")

        self.cutoff = self.lib.ace_omp_get_cutoff()
        self.set_num_threads(num_threads)

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        self.lib.ace_omp_init.restype = ctypes.c_int
        self.lib.ace_omp_init.argtypes = [ctypes.c_char_p]

        self.lib.ace_omp_get_num_threads.restype = ctypes.c_int
        self.lib.ace_omp_set_num_threads.argtypes = [ctypes.c_int]

        self.lib.ace_omp_get_cutoff.restype = ctypes.c_double

        self.lib.ace_omp_energy_forces_virial.restype = ctypes.c_double
        self.lib.ace_omp_energy_forces_virial.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]

    def get_num_threads(self) -> int:
        """Get number of OpenMP threads."""
        return self.lib.ace_omp_get_num_threads()

    def set_num_threads(self, n: int):
        """Set number of OpenMP threads."""
        self.lib.ace_omp_set_num_threads(n)

    def calculate(
        self,
        atoms: 'Atoms' = None,
        properties: list = None,
        system_changes: list = None
    ):
        """
        Calculate energy, forces, and stress.

        This is called by ASE when atoms.get_potential_energy() etc. are called.
        """
        if properties is None:
            properties = self.implemented_properties

        if HAS_ASE:
            Calculator.calculate(self, atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        # Get atomic data
        natoms = len(atoms)
        species = np.array(atoms.get_atomic_numbers(), dtype=np.int32)
        positions = np.array(atoms.get_positions(), dtype=np.float64).flatten()

        # Handle periodic boundary conditions
        pbc = atoms.get_pbc()
        if any(pbc):
            cell = np.array(atoms.get_cell(), dtype=np.float64).flatten()
            pbc_arr = np.array(pbc, dtype=np.int32)
        else:
            cell = None
            pbc_arr = None

        # Allocate outputs
        forces = np.zeros(natoms * 3, dtype=np.float64)
        virial = np.zeros(9, dtype=np.float64)

        # Call OpenMP wrapper
        energy = self.lib.ace_omp_energy_forces_virial(
            natoms,
            species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if cell is not None else None,
            pbc_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) if pbc_arr is not None else None,
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            virial.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        # Store results
        self.results['energy'] = energy
        self.results['forces'] = forces.reshape(natoms, 3)

        # Convert virial to stress (stress = -virial / volume)
        if any(pbc):
            volume = atoms.get_volume()
            # virial is 3x3 row-major, convert to Voigt notation for ASE
            virial_matrix = virial.reshape(3, 3)
            stress_voigt = np.array([
                virial_matrix[0, 0],  # xx
                virial_matrix[1, 1],  # yy
                virial_matrix[2, 2],  # zz
                virial_matrix[1, 2],  # yz
                virial_matrix[0, 2],  # xz
                virial_matrix[0, 1],  # xy
            ])
            self.results['stress'] = -stress_voigt / volume
        else:
            self.results['stress'] = np.zeros(6)


def benchmark(ace_lib_path: str, supercell_size: int = 3, num_threads_list: list = None):
    """
    Benchmark the OpenMP calculator.

    Args:
        ace_lib_path: Path to ACE library
        supercell_size: Size of diamond Si supercell (n x n x n)
        num_threads_list: List of thread counts to test
    """
    import time

    if not HAS_ASE:
        raise ImportError("ASE required for benchmark")

    from ase.build import bulk

    if num_threads_list is None:
        num_threads_list = [1, 2, 4, 8]

    # Create supercell
    atoms = bulk('Si', 'diamond', a=5.43) * supercell_size
    print(f"Benchmarking with {len(atoms)} atoms ({supercell_size}x{supercell_size}x{supercell_size} supercell)")

    calc = ACECalculatorOMP(ace_lib_path, num_threads=1)
    atoms.calc = calc

    results = {}
    for nt in num_threads_list:
        calc.set_num_threads(nt)

        # Warm up
        atoms.get_potential_energy()

        # Benchmark
        nreps = 5
        start = time.time()
        for _ in range(nreps):
            E = atoms.get_potential_energy()
            F = atoms.get_forces()
        elapsed = time.time() - start

        time_per_call = elapsed / nreps * 1000
        results[nt] = time_per_call

        speedup = results[1] / time_per_call if 1 in results else 1.0
        print(f"  {nt} threads: {time_per_call:.2f} ms/call, speedup={speedup:.2f}x")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ace_calculator_omp.py <ace_library.so> [supercell_size]")
        sys.exit(1)

    lib_path = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    benchmark(lib_path, size)
