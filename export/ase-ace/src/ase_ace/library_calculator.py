"""
ACE Calculator using compiled Julia shared library.

This module provides an ASE calculator that uses a pre-compiled ACE potential
shared library (.so file) instead of running Julia via sockets. This approach:
- Has instant startup (no JIT compilation)
- Requires a deployment package created by ACEpotentials.jl
- No Julia installation needed at runtime

Note: Multi-threading is NOT supported with compiled libraries (--trim=safe
removes the required closure specializations). For multi-threaded evaluation,
use ACECalculator (socket-based) or LAMMPS with OpenMP.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union

from ase.calculators.calculator import all_changes

from .base import ACECalculatorBase

# Try to import matscipy for efficient neighbor lists
try:
    from matscipy.neighbours import neighbour_list
    HAS_MATSCIPY = True
except ImportError:
    HAS_MATSCIPY = False


class ACELibrary:
    """
    Low-level wrapper for the ACE shared library using ctypes.

    This class provides direct access to the C functions exported
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

        # Load the shared library
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
        """Configure ctypes function signatures."""

        # Utility functions
        self.lib.ace_get_cutoff.restype = ctypes.c_double
        self.lib.ace_get_cutoff.argtypes = []

        self.lib.ace_get_n_species.restype = ctypes.c_int
        self.lib.ace_get_n_species.argtypes = []

        self.lib.ace_get_species.restype = ctypes.c_int
        self.lib.ace_get_species.argtypes = [ctypes.c_int]

        # Basis functions (for descriptor computation)
        self._setup_basis_functions()

        # Site-level functions (always available)
        self.lib.ace_site_energy_forces_virial.restype = ctypes.c_double
        self.lib.ace_site_energy_forces_virial.argtypes = [
            ctypes.c_int,                              # z0
            ctypes.c_int,                              # nneigh
            ctypes.POINTER(ctypes.c_int),              # neighbor_z
            ctypes.POINTER(ctypes.c_double),           # neighbor_Rij
            ctypes.POINTER(ctypes.c_double),           # forces (output)
            ctypes.POINTER(ctypes.c_double),           # virial (output)
        ]

        # Try to set up batch API (may not be available in --trim=safe builds)
        self.has_batch_api = False
        try:
            self.lib.ace_batch_energy_forces_virial.restype = None
            self.lib.ace_batch_energy_forces_virial.argtypes = [
                ctypes.c_int,                              # natoms
                ctypes.POINTER(ctypes.c_int),              # z
                ctypes.POINTER(ctypes.c_int),              # neighbor_counts
                ctypes.POINTER(ctypes.c_int),              # neighbor_offsets
                ctypes.POINTER(ctypes.c_int),              # neighbor_z
                ctypes.POINTER(ctypes.c_double),           # neighbor_Rij
                ctypes.POINTER(ctypes.c_double),           # energies (output)
                ctypes.POINTER(ctypes.c_double),           # forces (output)
                ctypes.POINTER(ctypes.c_double),           # virials (output)
            ]
            self.has_batch_api = True
        except AttributeError:
            pass  # Batch API not available, will use site-level API

    def _setup_basis_functions(self):
        """Configure ctypes function signatures for basis evaluation."""
        self.has_basis_api = False
        self.n_basis = 0

        try:
            self.lib.ace_get_n_basis.restype = ctypes.c_int
            self.lib.ace_get_n_basis.argtypes = []

            self.lib.ace_site_basis.restype = ctypes.c_int
            self.lib.ace_site_basis.argtypes = [
                ctypes.c_int,                    # z0
                ctypes.c_int,                    # nneigh
                ctypes.POINTER(ctypes.c_int),    # neighbor_z
                ctypes.POINTER(ctypes.c_double), # neighbor_Rij
                ctypes.POINTER(ctypes.c_double), # basis_out
            ]

            self.n_basis = self.lib.ace_get_n_basis()
            self.has_basis_api = True
        except AttributeError:
            pass  # Basis API not available (older library version)

    def site_basis(
        self,
        z0: int,
        neighbor_z: np.ndarray,
        neighbor_Rij: np.ndarray
    ) -> np.ndarray:
        """
        Compute basis (descriptors) for a single atom site.

        Parameters
        ----------
        z0 : int
            Atomic number of center atom
        neighbor_z : np.ndarray
            Atomic numbers of neighbor atoms
        neighbor_Rij : np.ndarray
            Displacement vectors to neighbors (nneigh, 3)

        Returns
        -------
        np.ndarray
            Basis vector of shape (n_basis,)
        """
        if not self.has_basis_api:
            raise RuntimeError(
                "Basis API not available. Ensure the library was compiled "
                "with ACEpotentials.jl v0.10.4 or later."
            )

        nneigh = len(neighbor_z)
        basis_out = np.zeros(self.n_basis, dtype=np.float64)

        if nneigh == 0:
            return basis_out

        neighbor_z = np.ascontiguousarray(neighbor_z, dtype=np.int32)
        neighbor_Rij = np.ascontiguousarray(neighbor_Rij, dtype=np.float64).flatten()

        self.lib.ace_site_basis(
            ctypes.c_int(z0),
            ctypes.c_int(nneigh),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_Rij.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            basis_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return basis_out

    def batch_energy_forces_virial(
        self,
        z: np.ndarray,
        neighbor_counts: np.ndarray,
        neighbor_offsets: np.ndarray,
        neighbor_z: np.ndarray,
        neighbor_Rij: np.ndarray
    ):
        """
        Compute energies, forces, and virials for multiple atoms.

        Note: Threading is NOT available in --trim=safe compiled libraries.
        """
        natoms = len(z)
        total_neighbors = len(neighbor_z)

        # Ensure contiguous arrays with correct dtypes
        z = np.ascontiguousarray(z, dtype=np.int32)
        neighbor_counts = np.ascontiguousarray(neighbor_counts, dtype=np.int32)
        neighbor_offsets = np.ascontiguousarray(neighbor_offsets, dtype=np.int32)
        neighbor_z = np.ascontiguousarray(neighbor_z, dtype=np.int32)
        neighbor_Rij = np.ascontiguousarray(neighbor_Rij, dtype=np.float64).flatten()

        # Output arrays
        energies = np.zeros(natoms, dtype=np.float64)
        forces = np.zeros(total_neighbors * 3, dtype=np.float64)
        virials = np.zeros(natoms * 6, dtype=np.float64)

        self.lib.ace_batch_energy_forces_virial(
            ctypes.c_int(natoms),
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_Rij.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            virials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energies, forces.reshape(total_neighbors, 3), virials.reshape(natoms, 6)

    def site_energy_forces_virial(
        self,
        z0: int,
        neighbor_z: np.ndarray,
        neighbor_Rij: np.ndarray
    ):
        """
        Compute energy, forces, and virial for a single atom site.

        This is the fallback when batch API is not available.
        """
        nneigh = len(neighbor_z)

        if nneigh == 0:
            return 0.0, np.zeros((0, 3), dtype=np.float64), np.zeros(6, dtype=np.float64)

        neighbor_z = np.ascontiguousarray(neighbor_z, dtype=np.int32)
        neighbor_Rij = np.ascontiguousarray(neighbor_Rij, dtype=np.float64).flatten()

        forces_out = np.zeros(nneigh * 3, dtype=np.float64)
        virial_out = np.zeros(6, dtype=np.float64)

        energy = self.lib.ace_site_energy_forces_virial(
            ctypes.c_int(z0),
            ctypes.c_int(nneigh),
            neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neighbor_Rij.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            forces_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            virial_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return energy, forces_out.reshape(-1, 3), virial_out


class ACELibraryCalculator(ACECalculatorBase):
    """
    ASE calculator using a pre-compiled ACE potential library.

    This calculator uses a compiled Julia ACE potential (.so file) with
    matscipy neighbor lists. It provides instant startup with no JIT delay.

    Note: Multi-threading is NOT supported (Julia --trim=safe removes required
    closures). For threaded evaluation, use ACECalculator or LAMMPS.

    Parameters
    ----------
    library_path : str
        Path to the compiled ACE shared library (.so file)
    **kwargs
        Additional arguments passed to ASE Calculator

    Example
    -------
    >>> from ase.build import bulk
    >>> from ase_ace import ACELibraryCalculator
    >>>
    >>> calc = ACELibraryCalculator("deployment/lib/libace_model.so")
    >>> atoms = bulk('Si', 'diamond', a=5.43)
    >>> atoms.calc = calc
    >>> print(f"Energy: {atoms.get_potential_energy():.4f} eV")
    """

    def __init__(self, library_path: str, **kwargs):
        if not HAS_MATSCIPY:
            raise ImportError(
                "matscipy is required for ACELibraryCalculator. "
                "Install it with: pip install matscipy"
            )

        super().__init__(**kwargs)
        self.ace = ACELibrary(library_path)

    @property
    def cutoff(self) -> float:
        """Cutoff radius in Angstroms."""
        return self.ace.cutoff

    @property
    def species(self) -> List[int]:
        """List of supported atomic numbers."""
        return self.ace.species

    @property
    def n_basis(self) -> int:
        """Number of basis functions (descriptors per atom)."""
        return self.ace.n_basis

    def _group_neighbors(self, i_idx: np.ndarray, natoms: int) -> tuple:
        """
        Group neighbor list by center atom.

        Parameters
        ----------
        i_idx : np.ndarray
            Center atom indices from neighbor list
        natoms : int
            Total number of atoms

        Returns
        -------
        tuple
            (atom_starts, atom_ends) arrays indicating neighbor ranges
        """
        if len(i_idx) > 0:
            atom_starts = np.searchsorted(i_idx, np.arange(natoms))
            atom_ends = np.searchsorted(i_idx, np.arange(natoms), side='right')
        else:
            atom_starts = np.zeros(natoms, dtype=int)
            atom_ends = np.zeros(natoms, dtype=int)
        return atom_starts, atom_ends

    def get_descriptors(
        self,
        atoms,
        include_forces: bool = False,
        include_virial: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute ACE descriptors (basis values) for all atoms.

        This method returns the raw ACE basis vectors for each atom,
        which can be used for:
        - Computing descriptors for fitting
        - Analysis and transfer learning
        - Verifying energy = descriptors @ weights (for linear models)

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic configuration
        include_forces : bool, default=False
            If True, also return per-basis force contributions.
            Note: Not yet supported for library calculator.
        include_virial : bool, default=False
            If True, also return per-basis virial contributions.
            Note: Not yet supported for library calculator.

        Returns
        -------
        np.ndarray or dict
            If include_forces=False and include_virial=False:
                Descriptor array of shape (natoms, n_basis)
            Otherwise:
                Dict with keys 'descriptors', 'energy', 'forces', 'virial'

        Example
        -------
        >>> calc = ACELibraryCalculator("libace.so")
        >>> atoms = bulk('Si', 'diamond', a=5.43)
        >>> D = calc.get_descriptors(atoms)
        >>> print(f"Descriptors shape: {D.shape}")  # (2, n_basis)
        """
        if include_forces or include_virial:
            raise NotImplementedError(
                "Force and virial descriptors are not yet supported in "
                "ACELibraryCalculator. Use ACEJuliaCalculator for this feature."
            )

        if not self.ace.has_basis_api:
            raise RuntimeError(
                "Descriptor API not available. Ensure the library was compiled "
                "with ACEpotentials.jl v0.10.4 or later."
            )

        natoms = len(atoms)
        numbers = atoms.get_atomic_numbers()

        # Build neighbor list using matscipy
        i_idx, j_idx, D_vectors = neighbour_list(
            'ijD',
            atoms,
            self.ace.cutoff
        )

        # Group neighbors by center atom
        atom_starts, atom_ends = self._group_neighbors(i_idx, natoms)

        # Compute descriptors for each atom
        descriptors = np.zeros((natoms, self.n_basis), dtype=np.float64)

        for i in range(natoms):
            z0 = int(numbers[i])
            start, end = atom_starts[i], atom_ends[i]
            nneigh = end - start

            if nneigh > 0:
                neigh_z = numbers[j_idx[start:end]]
                neigh_R = D_vectors[start:end]
            else:
                neigh_z = np.array([], dtype=np.int32)
                neigh_R = np.array([], dtype=np.float64).reshape(0, 3)

            descriptors[i, :] = self.ace.site_basis(z0, neigh_z, neigh_R)

        return descriptors

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Perform calculation for given atoms object.

        Uses matscipy neighbor list and site-level API.
        """
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        natoms = len(atoms)
        numbers = atoms.get_atomic_numbers()

        # Build neighbor list using matscipy (O(N) cell-list algorithm)
        i_idx, j_idx, D_vectors = neighbour_list(
            'ijD',
            atoms,
            self.ace.cutoff
        )

        # Group neighbors by center atom
        atom_starts, atom_ends = self._group_neighbors(i_idx, natoms)

        neighbor_counts = (atom_ends - atom_starts).astype(np.int32)

        if len(i_idx) == 0:
            # No neighbors - isolated atoms
            total_energy = 0.0
            forces = np.zeros((natoms, 3), dtype=np.float64)
            total_virial = np.zeros(6, dtype=np.float64)
        elif self.ace.has_batch_api:
            # Use batch API (faster, supports threading)
            z = numbers.astype(np.int32)
            neighbor_offsets = atom_starts.astype(np.int32)
            neighbor_z = numbers[j_idx].astype(np.int32)
            neighbor_Rij = np.ascontiguousarray(D_vectors, dtype=np.float64)

            energies, forces_on_neigh, virials = self.ace.batch_energy_forces_virial(
                z, neighbor_counts, neighbor_offsets, neighbor_z, neighbor_Rij
            )

            total_energy = np.sum(energies)
            total_virial = np.sum(virials, axis=0)

            # Accumulate forces using vectorized numpy operations (20x faster than loop)
            forces = np.zeros((natoms, 3), dtype=np.float64)
            np.add.at(forces, j_idx, forces_on_neigh)
            np.subtract.at(forces, i_idx, forces_on_neigh)
        else:
            # Use site-level API (fallback for --trim=safe builds)
            total_energy = 0.0
            total_virial = np.zeros(6, dtype=np.float64)
            forces_on_neigh = np.zeros((len(i_idx), 3), dtype=np.float64)

            for i in range(natoms):
                z0 = int(numbers[i])
                start, end = atom_starts[i], atom_ends[i]
                nneigh = end - start

                if nneigh > 0:
                    neigh_z = numbers[j_idx[start:end]]
                    neigh_R = D_vectors[start:end]

                    Ei, Fi, Vi = self.ace.site_energy_forces_virial(z0, neigh_z, neigh_R)
                    total_energy += Ei
                    total_virial += Vi
                    forces_on_neigh[start:end] = Fi

            # Accumulate forces using vectorized numpy operations
            forces = np.zeros((natoms, 3), dtype=np.float64)
            np.add.at(forces, j_idx, forces_on_neigh)
            np.subtract.at(forces, i_idx, forces_on_neigh)

        # Store results
        self.results['energy'] = total_energy
        self.results['forces'] = forces

        if 'stress' in properties:
            if atoms.pbc.any():
                volume = atoms.get_volume()
                stress = -total_virial / volume
                self.results['stress'] = stress
            else:
                self.results['stress'] = np.zeros(6)

    def __repr__(self) -> str:
        return f"ACELibraryCalculator(lib={self.ace.lib_path.name})"
