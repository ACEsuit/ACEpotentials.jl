"""
ASE Calculator using JuliaCall for direct Julia integration.

This module provides ACEJuliaCalculator, which uses JuliaCall to call
ACEpotentials.jl directly from Python. This enables multi-threading
and avoids the overhead of subprocess communication.

Requirements:
    pip install ase-ace[julia]

Or manually:
    pip install juliacall juliapkg
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
from ase.calculators.calculator import all_changes

from .base import ACECalculatorBase

logger = logging.getLogger(__name__)

# Path to Julia interface module (relative to this file)
_INTERFACE_PATH = Path(__file__).parent.parent.parent / "julia" / "python_interface.jl"


class ACEJuliaCalculator(ACECalculatorBase):
    """
    ASE calculator using JuliaCall for direct Julia integration.

    Uses Julia's energy_forces_virial() which handles neighbor lists
    and threading internally for maximum efficiency.

    Parameters
    ----------
    model_path : str
        Path to ACE model JSON file (created by ACEpotentials.save_model).
    num_threads : int or str, default='auto'
        Number of Julia threads. Set before first Julia initialization.
        Use 'auto' to use all available CPU cores.

    Examples
    --------
    >>> from ase.build import bulk
    >>> from ase_ace import ACEJuliaCalculator
    >>>
    >>> atoms = bulk('Si', 'diamond', a=5.43)
    >>> calc = ACEJuliaCalculator('model.json', num_threads=4)
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()

    Notes
    -----
    First calculation includes Julia JIT compilation (~10-30s).
    Subsequent calculations are fast (~ms).
    """

    # Class-level Julia state (shared across instances)
    _julia_initialized = False
    _jl = None

    def __init__(
        self,
        model_path: str,
        num_threads: Union[int, str] = 'auto',
    ):
        super().__init__()

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._num_threads = num_threads

        # Lazy initialization state
        self._model = None
        self._cutoff = None
        self._species = None
        self._n_basis = None

    @classmethod
    def _init_julia(cls, num_threads):
        """Initialize Julia runtime (once per process)."""
        if cls._julia_initialized:
            return cls._jl

        # Set thread count BEFORE importing juliacall
        if num_threads == 'auto':
            num_threads = os.cpu_count() or 1
        os.environ['JULIA_NUM_THREADS'] = str(num_threads)

        logger.info(f"Initializing Julia with {num_threads} threads...")

        # Import juliacall (this starts Julia)
        try:
            from juliacall import Main as jl
        except ImportError:
            raise ImportError(
                "juliacall not installed. Install with: pip install ase-ace[julia]"
            )

        # Load required Julia packages
        logger.info("Loading Julia packages...")
        jl.seval('using ACEpotentials')
        jl.seval('using AtomsBase')
        jl.seval('using AtomsCalculators')
        jl.seval('using Unitful')
        jl.seval('using StaticArrays')

        # Load Python interface module
        if _INTERFACE_PATH.exists():
            logger.info(f"Loading Python interface module: {_INTERFACE_PATH}")
            jl.seval(f'include("{_INTERFACE_PATH}")')
            jl.seval('using .ACEPythonInterface')
        else:
            raise FileNotFoundError(
                f"Julia interface module not found: {_INTERFACE_PATH}"
            )

        cls._jl = jl
        cls._julia_initialized = True

        logger.info(f"Julia initialized (version {jl.VERSION})")
        return jl

    def _ensure_initialized(self):
        """Lazy initialization of Julia and model loading."""
        if self._model is not None:
            return

        jl = self._init_julia(self._num_threads)

        logger.info(f"Loading model: {self.model_path}")

        # Load model via ACEpotentials.load_model
        result = jl.ACEpotentials.load_model(str(self.model_path))
        self._model = result[0]  # ACEPotential

        # Extract model properties
        self._cutoff = float(jl.Unitful.ustrip(
            jl.ACEpotentials.Models.cutoff_radius(self._model)
        ))

        # Get elements from model
        i2z = self._model.model._i2z
        self._species = [int(z) for z in i2z]

        # Get basis size
        self._n_basis = int(jl.ACEpotentials.Models.length_basis(self._model))

        logger.info(
            f"Model loaded: cutoff={self._cutoff:.2f} Å, "
            f"species={self._species}, n_basis={self._n_basis}"
        )

    @property
    def cutoff(self) -> float:
        """Cutoff radius in Angstroms."""
        self._ensure_initialized()
        return self._cutoff

    @property
    def species(self) -> List[int]:
        """List of supported atomic numbers."""
        self._ensure_initialized()
        return self._species

    @property
    def n_basis(self) -> int:
        """Number of basis functions per atom."""
        self._ensure_initialized()
        return self._n_basis

    def _ase_to_atomsbase(self, atoms):
        """Convert ASE Atoms to Julia AtomsBase system."""
        jl = self._jl
        iface = jl.ACEPythonInterface

        # Extract data from ASE Atoms
        positions = atoms.get_positions()  # Angstroms
        numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell().array
        pbc = atoms.get_pbc()

        # Convert to Julia types using interface module
        # Use Fortran order (column-major) to match Julia's reshape
        n_atoms = len(atoms)
        positions_jl = iface.make_positions(
            positions.T.flatten('F').tolist(), n_atoms
        )

        # Create cell vectors with units (as Vector of SVectors)
        # Cell rows in ASE are lattice vectors, so transpose and use F-order
        cell_jl = iface.make_cell(cell.T.flatten('F').tolist())

        # Create atoms with species
        atoms_jl = iface.make_atoms(positions_jl, numbers.tolist())

        # Create periodic system with boundary conditions
        pbc_tuple = tuple(bool(p) for p in pbc)
        system_jl = iface.make_system(atoms_jl, cell_jl, pbc_tuple)

        return system_jl

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes
    ):
        """
        Perform calculation for the given atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            ASE Atoms object to calculate.
        properties : list of str
            Properties to calculate ('energy', 'forces', 'stress').
        system_changes : list of str
            Changes since last calculation.
        """
        if properties is None:
            properties = self.implemented_properties

        # Call parent to set self.atoms
        super().calculate(atoms, properties, system_changes)

        self._ensure_initialized()
        jl = self._jl
        iface = jl.ACEPythonInterface

        # Convert ASE Atoms to AtomsBase
        at_jl = self._ase_to_atomsbase(atoms)

        # Call Julia's energy_forces_virial
        result = jl.AtomsCalculators.energy_forces_virial(at_jl, self._model)

        # Extract energy (strip units)
        self.results['energy'] = float(jl.Unitful.ustrip(result.energy))

        # Extract forces using interface module
        self.results['forces'] = np.array(iface.extract_forces(result.forces))

        # Extract virial and convert to stress
        if atoms.pbc.any():
            virial = np.array(iface.extract_virial(result.virial))
            volume = atoms.get_volume()

            # Convert to Voigt notation stress: xx, yy, zz, yz, xz, xy
            self.results['stress'] = np.array([
                -virial[0, 0] / volume,
                -virial[1, 1] / volume,
                -virial[2, 2] / volume,
                -virial[1, 2] / volume,
                -virial[0, 2] / volume,
                -virial[0, 1] / volume,
            ])
        else:
            self.results['stress'] = np.zeros(6)

    def get_descriptors(
        self,
        atoms,
        include_forces: bool = False,
        include_virial: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute ACE descriptors (basis values) for all atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic configuration
        include_forces : bool, default=False
            If True, also return per-basis force contributions
        include_virial : bool, default=False
            If True, also return per-basis virial contributions

        Returns
        -------
        np.ndarray or dict
            If include_forces=False and include_virial=False:
                Descriptor array of shape (natoms, n_basis)
            Otherwise:
                Dict with keys:
                - 'descriptors': (natoms, n_basis) site energy descriptors
                - 'energy': (n_basis,) per-basis energy contributions
                - 'forces': (natoms, n_basis, 3) per-basis force contributions (if requested)
                - 'virial': (n_basis, 3, 3) per-basis virial contributions (if requested)
        """
        self._ensure_initialized()
        jl = self._jl
        iface = jl.ACEPythonInterface

        # Convert ASE Atoms to AtomsBase
        at_jl = self._ase_to_atomsbase(atoms)

        if not include_forces and not include_virial:
            # Use site_descriptors for energy descriptors only
            descriptors_jl = iface.compute_site_descriptors(at_jl, self._model)
            return np.array(descriptors_jl)
        else:
            # Use energy_forces_virial_basis for full descriptor set
            descriptors = np.array(
                iface.compute_site_descriptors(at_jl, self._model)
            )
            result = iface.compute_force_virial_descriptors(at_jl, self._model)

            output = {
                'descriptors': descriptors,
                'energy': np.array(result.energy),
            }
            if include_forces:
                output['forces'] = np.array(result.forces)
            if include_virial:
                output['virial'] = np.array(result.virial)

            return output

    def __repr__(self) -> str:
        return f"ACEJuliaCalculator(model={self.model_path.name}, threads={self._num_threads})"
