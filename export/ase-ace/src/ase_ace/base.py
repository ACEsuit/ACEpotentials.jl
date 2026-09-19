"""
Abstract base class for ACE calculators.

This module defines ACECalculatorBase, which provides a unified API
for all ACE calculator backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union

import numpy as np
from ase.calculators.calculator import Calculator


class ACECalculatorBase(Calculator, ABC):
    """
    Abstract base class for ACE calculators.

    Provides a unified API across different backends:
    - ACELibraryCalculator (compiled .so via ctypes) - full support
    - ACEJuliaCalculator (JuliaCall) - full support
    - ACECalculator (socket-based) - energy/forces/stress only, no descriptors

    Subclasses must implement the abstract properties and methods.

    Properties
    ----------
    cutoff : float
        Cutoff radius in Angstroms.
    species : List[int]
        List of supported atomic numbers.
    n_basis : int
        Number of basis functions per atom.

    Methods
    -------
    get_descriptors(atoms)
        Compute ACE descriptors (basis values) for all atoms.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    @property
    @abstractmethod
    def cutoff(self) -> float:
        """Cutoff radius in Angstroms."""
        pass

    @property
    @abstractmethod
    def species(self) -> List[int]:
        """List of supported atomic numbers."""
        pass

    @property
    @abstractmethod
    def n_basis(self) -> int:
        """Number of basis functions per atom."""
        pass

    @abstractmethod
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
        pass
