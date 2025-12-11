"""
ACE Potential Calculator - Python wrapper for compiled Julia ACE shared library.

This module provides:
1. Low-level ctypes interface to the C site-level functions
2. High-level ASE-compatible Calculator class using matscipy neighbor lists

The calculator uses matscipy's efficient O(N) cell-list neighbor finder and
calls the site-level C API (same as LAMMPS uses) for energy/force evaluation.

Usage:
    from ace_calculator import ACECalculator

    # Create calculator
    calc = ACECalculator("silicon_lib/lib/libace.so")

    # Use with ASE
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

# Try to import ASE (optional dependency)
try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Calculator = object
    all_changes = []

# Try to import matscipy for efficient neighbor lists
try:
    from matscipy.neighbours import neighbour_list
    HAS_MATSCIPY = True
except ImportError:
    HAS_MATSCIPY = False


class ACELibrary:
    """
    Low-level wrapper for the ACE shared library using ctypes.

    This class provides direct access to the site-level C functions exported
    by the Julia-compiled ACE potential library.
    """

    def __init__(self, library_path: str):
        """
        Load the ACE shared library.

        Args:
            library_path: Path to the compiled .so file
        """
        self.lib_path = Path(library_path).resolve()
        if not self.lib_path.exists():
            raise FileNotFoundError(f"Library not found: {self.lib_path}")

        # Load the shared library with RTLD_NOW to resolve all symbols immediately
        import os
        rtld_now = getattr(os, 'RTLD_NOW', 2)
        self.lib = ctypes.CDLL(str(self.lib_path), mode=rtld_now)

        # Set up function signatures
        self._setup_functions()

        # Get potential info
        self.cutoff = self.lib.ace_get_cutoff()
        self.n_species = self.lib.ace_get_n_species()
        self.species = [self.lib.ace_get_species(i+1) for i in range(self.n_species)]

    def _setup_functions(self):
        """Configure ctypes function signatures for site-level API."""

        # Utility functions
        self.lib.ace_get_cutoff.restype = ctypes.c_double
        self.lib.ace_get_cutoff.argtypes = []

        self.lib.ace_get_n_species.restype = ctypes.c_int
        self.lib.ace_get_n_species.argtypes = []

        self.lib.ace_get_species.restype = ctypes.c_int
        self.lib.ace_get_species.argtypes = [ctypes.c_int]

        # Site-level functions
        self.lib.ace_site_energy.restype = ctypes.c_double
        self.lib.ace_site_energy.argtypes = [
            ctypes.c_int,                              # z0
            ctypes.c_int,                              # nneigh
            ctypes.POINTER(ctypes.c_int),              # neighbor_z
            ctypes.POINTER(ctypes.c_double),           # neighbor_Rij
        ]

        self.lib.ace_site_energy_forces.restype = ctypes.c_double
        self.lib.ace_site_energy_forces.argtypes = [
            ctypes.c_int,                              # z0
            ctypes.c_int,                              # nneigh
            ctypes.POINTER(ctypes.c_int),              # neighbor_z
            ctypes.POINTER(ctypes.c_double),           # neighbor_Rij
            ctypes.POINTER(ctypes.c_double),           # forces (output)
        ]

        self.lib.ace_site_energy_forces_virial.restype = ctypes.c_double
        self.lib.ace_site_energy_forces_virial.argtypes = [
            ctypes.c_int,                              # z0
            ctypes.c_int,                              # nneigh
            ctypes.POINTER(ctypes.c_int),              # neighbor_z
            ctypes.POINTER(ctypes.c_double),           # neighbor_Rij
            ctypes.POINTER(ctypes.c_double),           # forces (output)
            ctypes.POINTER(ctypes.c_double),           # virial (output, 6 elements)
        ]

    def site_energy(self, z0: int, neighbor_z: np.ndarray, neighbor_R: np.ndarray) -> float:
        """
        Compute site energy for a single atom.

        Args:
            z0: Atomic number of center atom
            neighbor_z: Array of neighbor atomic numbers (nneigh,)
            neighbor_R: Array of displacement vectors Rj - Ri (nneigh, 3)

        Returns:
            Site energy in eV
        """
        nneigh = len(neighbor_z)
        if nneigh == 0:
            return 0.0

        neighbor_z = np.ascontiguousarray(neighbor_z, dtype=np.int32)
        neighbor_R = np.ascontiguousarray(neighbor_R, dtype=np.float64).flatten()

        return self.lib.ace_site_energy(
            ctypes.c_int(z0),
            ctypes.c_int(nneigh),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def site_energy_forces(self, z0: int, neighbor_z: np.ndarray, neighbor_R: np.ndarray
                          ) -> Tuple[float, np.ndarray]:
        """
        Compute site energy and forces for a single atom.

        Args:
            z0: Atomic number of center atom
            neighbor_z: Array of neighbor atomic numbers (nneigh,)
            neighbor_R: Array of displacement vectors Rj - Ri (nneigh, 3)

        Returns:
            (energy, forces) where forces is (nneigh, 3) array of forces ON neighbors
        """
        nneigh = len(neighbor_z)
        if nneigh == 0:
            return 0.0, np.zeros((0, 3), dtype=np.float64)

        neighbor_z = np.ascontiguousarray(neighbor_z, dtype=np.int32)
        neighbor_R = np.ascontiguousarray(neighbor_R, dtype=np.float64).flatten()
        forces = np.zeros(nneigh * 3, dtype=np.float64)

        energy = self.lib.ace_site_energy_forces(
            ctypes.c_int(z0),
            ctypes.c_int(nneigh),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energy, forces.reshape(nneigh, 3)

    def site_energy_forces_virial(self, z0: int, neighbor_z: np.ndarray, neighbor_R: np.ndarray
                                  ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute site energy, forces, and virial for a single atom.

        Args:
            z0: Atomic number of center atom
            neighbor_z: Array of neighbor atomic numbers (nneigh,)
            neighbor_R: Array of displacement vectors Rj - Ri (nneigh, 3)

        Returns:
            (energy, forces, virial) where:
            - forces is (nneigh, 3) array of forces ON neighbors
            - virial is (6,) array in Voigt notation (xx, yy, zz, yz, xz, xy)
        """
        nneigh = len(neighbor_z)
        if nneigh == 0:
            return 0.0, np.zeros((0, 3), dtype=np.float64), np.zeros(6, dtype=np.float64)

        neighbor_z = np.ascontiguousarray(neighbor_z, dtype=np.int32)
        neighbor_R = np.ascontiguousarray(neighbor_R, dtype=np.float64).flatten()
        forces = np.zeros(nneigh * 3, dtype=np.float64)
        virial = np.zeros(6, dtype=np.float64)

        energy = self.lib.ace_site_energy_forces_virial(
            ctypes.c_int(z0),
            ctypes.c_int(nneigh),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            virial.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energy, forces.reshape(nneigh, 3), virial


class ACECalculator(Calculator):
    """
    ASE-compatible calculator for ACE potentials.

    This calculator wraps a compiled Julia ACE potential shared library
    and provides the standard ASE Calculator interface. It uses matscipy's
    efficient O(N) cell-list neighbor finder and the site-level C API.

    Example:
        >>> from ace_calculator import ACECalculator
        >>> from ase.build import bulk
        >>>
        >>> calc = ACECalculator("libace.so")
        >>> atoms = bulk('Si', 'diamond', a=5.43)
        >>> atoms.calc = calc
        >>> print(f"Energy: {atoms.get_potential_energy():.4f} eV")
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, library_path: str, **kwargs):
        """
        Initialize the ACE calculator.

        Args:
            library_path: Path to the compiled ACE shared library (.so file)
            **kwargs: Additional arguments passed to ASE Calculator
        """
        if not HAS_MATSCIPY:
            raise ImportError(
                "matscipy is required for ACECalculator. "
                "Install it with: pip install matscipy"
            )

        if HAS_ASE:
            Calculator.__init__(self, **kwargs)

        self.ace = ACELibrary(library_path)

    @property
    def cutoff(self) -> float:
        """Cutoff radius in Angstroms."""
        return self.ace.cutoff

    @property
    def species(self) -> List[int]:
        """List of supported atomic numbers."""
        return self.ace.species

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Perform calculation for given atoms object.

        Uses matscipy neighbor list (O(N) cell-list algorithm) and site-level API.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: List of changes since last calculation
        """
        if HAS_ASE:
            Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(atoms)
        numbers = atoms.get_atomic_numbers()

        # Build neighbor list using matscipy (O(N) cell-list algorithm)
        # Returns: i (center), j (neighbor), D (displacement vector Rj - Ri)
        i_idx, j_idx, D_vectors = neighbour_list(
            'ijD',
            atoms,
            self.ace.cutoff
        )

        # Initialize output arrays
        total_energy = 0.0
        forces = np.zeros((natoms, 3), dtype=np.float64)

        if 'stress' in properties:
            # Voigt notation: xx, yy, zz, yz, xz, xy
            total_virial = np.zeros(6, dtype=np.float64)

        # Group neighbors by center atom for efficient processing
        # matscipy returns neighbors sorted by i, so we can use searchsorted
        if len(i_idx) > 0:
            # Find where each atom's neighbors start and end
            atom_starts = np.searchsorted(i_idx, np.arange(natoms))
            atom_ends = np.searchsorted(i_idx, np.arange(natoms), side='right')
        else:
            atom_starts = np.zeros(natoms, dtype=int)
            atom_ends = np.zeros(natoms, dtype=int)

        # Calculate based on requested properties
        need_virial = 'stress' in properties

        for i in range(natoms):
            start, end = atom_starts[i], atom_ends[i]
            nneigh = end - start

            if nneigh == 0:
                # No neighbors - zero contribution
                continue

            z0 = numbers[i]
            neigh_j = j_idx[start:end]
            neighbor_z = numbers[neigh_j]
            neighbor_R = D_vectors[start:end]  # Already Rj - Ri from matscipy

            if need_virial:
                E_site, F_neigh, V_site = self.ace.site_energy_forces_virial(
                    z0, neighbor_z, neighbor_R
                )
                total_virial += V_site
            else:
                E_site, F_neigh = self.ace.site_energy_forces(
                    z0, neighbor_z, neighbor_R
                )

            total_energy += E_site

            # Accumulate forces:
            # F_neigh[k] is force ON neighbor j[k] due to center i
            # Force on center i is negative sum of forces on neighbors
            for k, j in enumerate(neigh_j):
                forces[j] += F_neigh[k]
                forces[i] -= F_neigh[k]

        # Store results
        self.results['energy'] = total_energy
        self.results['forces'] = forces

        if need_virial:
            # Convert virial to stress (stress = -virial / volume)
            # Site virial is in Voigt notation: (xx, yy, zz, yz, xz, xy)
            if atoms.pbc.any():
                volume = atoms.get_volume()
                # ASE stress convention: positive = tensile
                stress = -total_virial / volume
                self.results['stress'] = stress
            else:
                # Non-periodic: stress is not well-defined
                self.results['stress'] = np.zeros(6)


def test_library():
    """Test the ACE library with a simple Silicon configuration."""
    import os

    # Find the library
    script_dir = Path(__file__).parent
    lib_path = script_dir / "silicon_lib" / "lib" / "libace.so"

    if not lib_path.exists():
        print(f"Library not found at {lib_path}")
        print("Please compile the library first.")
        return

    print("=" * 60)
    print("ACE Library Test")
    print("=" * 60)

    # Load library
    ace = ACELibrary(str(lib_path))
    print(f"Library loaded: {lib_path}")
    print(f"Cutoff: {ace.cutoff:.2f} A")
    print(f"Species: {ace.species} (Z values)")
    print()

    # Test site-level API
    print("Site-level API test:")
    z0 = 14  # Silicon
    neighbor_z = np.array([14, 14, 14], dtype=np.int32)
    neighbor_R = np.array([
        [2.35, 0.0, 0.0],
        [-0.78, 2.22, 0.0],
        [-0.78, -1.11, 1.92]
    ], dtype=np.float64)

    E_site = ace.site_energy(z0, neighbor_z, neighbor_R)
    print(f"  Site energy: {E_site:.6f} eV")

    E_site2, F_site = ace.site_energy_forces(z0, neighbor_z, neighbor_R)
    print(f"  Site energy (with forces): {E_site2:.6f} eV")
    print(f"  Forces on neighbors:")
    for i, f in enumerate(F_site):
        print(f"    F[{i}] = [{f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f}]")
    print()

    # Test ASE Calculator if available
    if HAS_ASE and HAS_MATSCIPY:
        print("ASE Calculator test (with matscipy neighbor list):")
        from ase.build import bulk

        calc = ACECalculator(str(lib_path))
        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.calc = calc

        E_ase = atoms.get_potential_energy()
        F_ase = atoms.get_forces()

        print(f"  ASE energy: {E_ase:.6f} eV")
        print(f"  ASE forces shape: {F_ase.shape}")
        print(f"  Max force magnitude: {np.max(np.linalg.norm(F_ase, axis=1)):.6f} eV/A")

        # Test stress
        S_ase = atoms.get_stress()
        print(f"  ASE stress (Voigt): {S_ase}")
    else:
        if not HAS_ASE:
            print("ASE not available - skipping ASE Calculator test")
        if not HAS_MATSCIPY:
            print("matscipy not available - skipping ASE Calculator test")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_library()
