"""
ACE Potential Calculator - Python wrapper for compiled Julia ACE shared library.

This module provides:
1. Low-level ctypes interface to the C functions
2. High-level ASE-compatible Calculator class

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


class ACELibrary:
    """
    Low-level wrapper for the ACE shared library using ctypes.

    This class provides direct access to the C functions exported by the
    Julia-compiled ACE potential library.
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
        # This ensures errors are caught at load time rather than delayed
        # Use os.RTLD_NOW for cross-platform compatibility (ctypes.RTLD_NOW not always available)
        import os
        rtld_now = getattr(os, 'RTLD_NOW', 2)  # 2 is the POSIX value for RTLD_NOW
        self.lib = ctypes.CDLL(str(self.lib_path), mode=rtld_now)

        # Set up function signatures
        self._setup_functions()

        # Get potential info
        self.cutoff = self.lib.ace_get_cutoff()
        self.n_species = self.lib.ace_get_n_species()
        self.species = [self.lib.ace_get_species(i+1) for i in range(self.n_species)]

    def _setup_functions(self):
        """Configure ctypes function signatures."""

        # Utility functions
        self.lib.ace_get_cutoff.restype = ctypes.c_double
        self.lib.ace_get_cutoff.argtypes = []

        self.lib.ace_get_n_species.restype = ctypes.c_int
        self.lib.ace_get_n_species.argtypes = []

        self.lib.ace_get_species.restype = ctypes.c_int
        self.lib.ace_get_species.argtypes = [ctypes.c_int]

        # Site-level functions (for LAMMPS)
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

        # System-level functions (for Python/ASE)
        self.lib.ace_energy.restype = ctypes.c_double
        self.lib.ace_energy.argtypes = [
            ctypes.c_int,                              # natoms
            ctypes.POINTER(ctypes.c_int),              # species
            ctypes.POINTER(ctypes.c_double),           # positions
            ctypes.POINTER(ctypes.c_double),           # cell (or NULL)
            ctypes.POINTER(ctypes.c_int),              # pbc (or NULL)
        ]

        self.lib.ace_energy_forces.restype = ctypes.c_double
        self.lib.ace_energy_forces.argtypes = [
            ctypes.c_int,                              # natoms
            ctypes.POINTER(ctypes.c_int),              # species
            ctypes.POINTER(ctypes.c_double),           # positions
            ctypes.POINTER(ctypes.c_double),           # cell (or NULL)
            ctypes.POINTER(ctypes.c_int),              # pbc (or NULL)
            ctypes.POINTER(ctypes.c_double),           # forces (output)
        ]

        self.lib.ace_energy_forces_virial.restype = ctypes.c_double
        self.lib.ace_energy_forces_virial.argtypes = [
            ctypes.c_int,                              # natoms
            ctypes.POINTER(ctypes.c_int),              # species
            ctypes.POINTER(ctypes.c_double),           # positions
            ctypes.POINTER(ctypes.c_double),           # cell (or NULL)
            ctypes.POINTER(ctypes.c_int),              # pbc (or NULL)
            ctypes.POINTER(ctypes.c_double),           # forces (output)
            ctypes.POINTER(ctypes.c_double),           # virial (output, 9 elements)
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
        neighbor_z = np.asarray(neighbor_z, dtype=np.int32)
        neighbor_R = np.asarray(neighbor_R, dtype=np.float64).flatten()

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
        neighbor_z = np.asarray(neighbor_z, dtype=np.int32)
        neighbor_R = np.asarray(neighbor_R, dtype=np.float64).flatten()
        forces = np.zeros(nneigh * 3, dtype=np.float64)

        energy = self.lib.ace_site_energy_forces(
            ctypes.c_int(z0),
            ctypes.c_int(nneigh),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energy, forces.reshape(nneigh, 3)

    def energy(self, species: np.ndarray, positions: np.ndarray,
               cell: Optional[np.ndarray] = None, pbc: Optional[np.ndarray] = None) -> float:
        """
        Compute total energy of a system.

        Args:
            species: Array of atomic numbers (natoms,)
            positions: Array of positions (natoms, 3)
            cell: Unit cell vectors (3, 3) or None for non-periodic
            pbc: Periodic boundary conditions (3,) or None

        Returns:
            Total energy in eV
        """
        natoms = len(species)
        species = np.asarray(species, dtype=np.int32)
        positions = np.asarray(positions, dtype=np.float64).flatten()

        cell_ptr = None
        pbc_ptr = None

        if cell is not None:
            cell = np.asarray(cell, dtype=np.float64).flatten()
            cell_ptr = cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        if pbc is not None:
            pbc = np.asarray(pbc, dtype=np.int32)
            pbc_ptr = pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        return self.lib.ace_energy(
            ctypes.c_int(natoms),
            species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cell_ptr,
            pbc_ptr,
        )

    def energy_forces(self, species: np.ndarray, positions: np.ndarray,
                      cell: Optional[np.ndarray] = None, pbc: Optional[np.ndarray] = None
                     ) -> Tuple[float, np.ndarray]:
        """
        Compute total energy and forces.

        Args:
            species: Array of atomic numbers (natoms,)
            positions: Array of positions (natoms, 3)
            cell: Unit cell vectors (3, 3) or None
            pbc: Periodic boundary conditions (3,) or None

        Returns:
            (energy, forces) tuple
        """
        natoms = len(species)
        species = np.asarray(species, dtype=np.int32)
        positions = np.asarray(positions, dtype=np.float64).flatten()
        forces = np.zeros(natoms * 3, dtype=np.float64)

        cell_ptr = None
        pbc_ptr = None

        if cell is not None:
            cell = np.asarray(cell, dtype=np.float64).flatten()
            cell_ptr = cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        if pbc is not None:
            pbc = np.asarray(pbc, dtype=np.int32)
            pbc_ptr = pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        energy = self.lib.ace_energy_forces(
            ctypes.c_int(natoms),
            species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cell_ptr,
            pbc_ptr,
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energy, forces.reshape(natoms, 3)

    def energy_forces_virial(self, species: np.ndarray, positions: np.ndarray,
                             cell: Optional[np.ndarray] = None, pbc: Optional[np.ndarray] = None
                            ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute total energy, forces, and virial stress.

        Args:
            species: Array of atomic numbers (natoms,)
            positions: Array of positions (natoms, 3)
            cell: Unit cell vectors (3, 3) or None
            pbc: Periodic boundary conditions (3,) or None

        Returns:
            (energy, forces, virial) tuple where virial is (3, 3)
        """
        natoms = len(species)
        species = np.asarray(species, dtype=np.int32)
        positions = np.asarray(positions, dtype=np.float64).flatten()
        forces = np.zeros(natoms * 3, dtype=np.float64)
        virial = np.zeros(9, dtype=np.float64)

        cell_ptr = None
        pbc_ptr = None

        if cell is not None:
            cell = np.asarray(cell, dtype=np.float64).flatten()
            cell_ptr = cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        if pbc is not None:
            pbc = np.asarray(pbc, dtype=np.int32)
            pbc_ptr = pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        energy = self.lib.ace_energy_forces_virial(
            ctypes.c_int(natoms),
            species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cell_ptr,
            pbc_ptr,
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            virial.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energy, forces.reshape(natoms, 3), virial.reshape(3, 3)


class ACECalculator(Calculator):
    """
    ASE-compatible calculator for ACE potentials.

    This calculator wraps a compiled Julia ACE potential shared library
    and provides the standard ASE Calculator interface.

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

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: List of changes since last calculation
        """
        if HAS_ASE:
            Calculator.calculate(self, atoms, properties, system_changes)

        # Get atomic data
        species = atoms.get_atomic_numbers()
        positions = atoms.get_positions()

        # Handle periodic boundary conditions
        if atoms.pbc.any():
            cell = atoms.get_cell()
            pbc = atoms.pbc.astype(np.int32)
        else:
            cell = None
            pbc = None

        # Calculate based on requested properties
        if 'stress' in properties or 'virial' in properties:
            energy, forces, virial = self.ace.energy_forces_virial(
                species, positions, cell, pbc
            )
            # Convert virial to stress (stress = -virial / volume)
            if atoms.pbc.any():
                volume = atoms.get_volume()
                stress = -virial / volume
                # Convert to Voigt notation for ASE: xx, yy, zz, yz, xz, xy
                self.results['stress'] = np.array([
                    stress[0, 0], stress[1, 1], stress[2, 2],
                    stress[1, 2], stress[0, 2], stress[0, 1]
                ])
            self.results['energy'] = energy
            self.results['forces'] = forces

        elif 'forces' in properties:
            energy, forces = self.ace.energy_forces(species, positions, cell, pbc)
            self.results['energy'] = energy
            self.results['forces'] = forces

        else:
            energy = self.ace.energy(species, positions, cell, pbc)
            self.results['energy'] = energy


def test_library():
    """Test the ACE library with a simple Silicon configuration."""
    import os

    # Find the library
    script_dir = Path(__file__).parent
    lib_path = script_dir / "silicon_lib" / "lib" / "libace.so"

    if not lib_path.exists():
        print(f"Library not found at {lib_path}")
        print("Please compile the library first with:")
        print("  julia scripts/trim_test/test_library_export.jl")
        print("  juliac --output-lib ...")
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

    # Test system-level API
    print("System-level API test:")
    # Diamond Si unit cell
    a = 5.43
    species = np.array([14, 14], dtype=np.int32)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [a/4, a/4, a/4]
    ], dtype=np.float64)
    cell = np.array([
        [a/2, a/2, 0],
        [a/2, 0, a/2],
        [0, a/2, a/2]
    ], dtype=np.float64)
    pbc = np.array([1, 1, 1], dtype=np.int32)

    E_total = ace.energy(species, positions, cell, pbc)
    print(f"  Total energy (2-atom cell): {E_total:.6f} eV")

    E_total2, forces = ace.energy_forces(species, positions, cell, pbc)
    print(f"  Total energy (with forces): {E_total2:.6f} eV")
    print(f"  Forces:")
    for i, f in enumerate(forces):
        print(f"    F[{i}] = [{f[0]:.6f}, {f[1]:.6f}, {f[2]:.6f}]")

    E_total3, forces3, virial = ace.energy_forces_virial(species, positions, cell, pbc)
    print(f"  Virial tensor:")
    for row in virial:
        print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
    print()

    # Test ASE Calculator if available
    if HAS_ASE:
        print("ASE Calculator test:")
        from ase.build import bulk

        calc = ACECalculator(str(lib_path))
        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.calc = calc

        E_ase = atoms.get_potential_energy()
        F_ase = atoms.get_forces()

        print(f"  ASE energy: {E_ase:.6f} eV")
        print(f"  ASE forces shape: {F_ase.shape}")
        print(f"  Max force magnitude: {np.max(np.linalg.norm(F_ase, axis=1)):.6f} eV/A")
    else:
        print("ASE not available - skipping ASE Calculator test")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_library()
