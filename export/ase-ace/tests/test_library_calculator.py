"""
Unit tests for ACELibraryCalculator.

Run with:
    pytest -v tests/test_library_calculator.py

Requires:
    - Compiled ACE library (.so file)
    - Set ACE_TEST_LIBRARY environment variable to library path
    - pip install ase-ace[library]
"""

import os
import pytest
import numpy as np
from pathlib import Path


# Get library path from environment or skip
TEST_LIBRARY = os.environ.get('ACE_TEST_LIBRARY')
if TEST_LIBRARY:
    TEST_LIBRARY = Path(TEST_LIBRARY)




@pytest.fixture(scope="session")
def library_path():
    """Path to compiled ACE library for testing."""
    if not TEST_LIBRARY or not TEST_LIBRARY.exists():
        pytest.skip(
            f"Test library not found. Set ACE_TEST_LIBRARY environment variable. "
            f"Current value: {TEST_LIBRARY}"
        )
    return str(TEST_LIBRARY)


@pytest.fixture(scope="session")
def library_calculator(library_path):
    """Create an ACELibraryCalculator for testing."""
    try:
        from ase_ace import ACELibraryCalculator
    except ImportError:
        pytest.skip("matscipy not installed. Run: pip install ase-ace[library]")

    return ACELibraryCalculator(library_path)


@pytest.fixture
def test_structure(library_calculator):
    """Create a test structure using species from the loaded library."""
    from ase.build import bulk
    from ase.data import chemical_symbols

    # Get the first species from the library
    z = library_calculator.species[0]
    symbol = chemical_symbols[z]

    # Use appropriate lattice constants for common elements
    lattice_params = {
        'Si': ('diamond', 5.43),
        'Ti': ('hcp', 2.95),  # a parameter for hcp
        'Al': ('fcc', 4.05),
        'Cu': ('fcc', 3.61),
        'Fe': ('bcc', 2.87),
    }

    if symbol in lattice_params:
        crystal, a = lattice_params[symbol]
        return bulk(symbol, crystal, a=a)
    else:
        # Default to fcc with estimated lattice constant
        return bulk(symbol, 'fcc', a=4.0)


@pytest.fixture
def test_supercell(library_calculator):
    """Create a test supercell using species from the loaded library."""
    from ase.build import bulk
    from ase.data import chemical_symbols

    z = library_calculator.species[0]
    symbol = chemical_symbols[z]

    lattice_params = {
        'Si': ('diamond', 5.43),
        'Ti': ('hcp', 2.95),
        'Al': ('fcc', 4.05),
        'Cu': ('fcc', 3.61),
        'Fe': ('bcc', 2.87),
    }

    if symbol in lattice_params:
        crystal, a = lattice_params[symbol]
        atoms = bulk(symbol, crystal, a=a)
    else:
        atoms = bulk(symbol, 'fcc', a=4.0)

    return atoms * (2, 2, 2)


@pytest.fixture
def perturbed_structure(library_calculator):
    """Create a perturbed test structure."""
    from ase.build import bulk
    from ase.data import chemical_symbols

    z = library_calculator.species[0]
    symbol = chemical_symbols[z]

    lattice_params = {
        'Si': ('diamond', 5.43),
        'Ti': ('hcp', 2.95),
        'Al': ('fcc', 4.05),
        'Cu': ('fcc', 3.61),
        'Fe': ('bcc', 2.87),
    }

    if symbol in lattice_params:
        crystal, a = lattice_params[symbol]
        atoms = bulk(symbol, crystal, a=a)
    else:
        atoms = bulk(symbol, 'fcc', a=4.0)

    atoms = atoms * (2, 2, 2)
    np.random.seed(42)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
    return atoms


class TestLibraryImports:
    """Test that the library calculator can be imported."""

    def test_import_library_calculator(self):
        """Test importing ACELibraryCalculator."""
        from ase_ace import ACELibraryCalculator
        assert ACELibraryCalculator is not None

    def test_import_requires_matscipy(self):
        """Test that matscipy is required."""
        from ase_ace.library_calculator import HAS_MATSCIPY
        # This will be True if matscipy is installed


class TestLibraryInit:
    """Test library calculator initialization."""

    def test_init_missing_library(self):
        """Test that missing library raises FileNotFoundError."""
        try:
            from ase_ace import ACELibraryCalculator
        except ImportError:
            pytest.skip("matscipy not installed")

        with pytest.raises(FileNotFoundError):
            ACELibraryCalculator('/nonexistent/libace.so')

    @pytest.mark.requires_library
    def test_init_properties(self, library_calculator):
        """Test calculator properties after init."""
        assert library_calculator.cutoff > 0
        assert len(library_calculator.species) > 0
        # Species depend on the model - just verify we got some valid atomic numbers
        for z in library_calculator.species:
            assert 1 <= z <= 118  # Valid atomic number range


@pytest.mark.requires_library
class TestLibraryCalculations:
    """Test actual calculations with compiled library."""

    def test_energy_finite(self, library_calculator, test_structure):
        """Test that energy is finite."""
        test_structure.calc = library_calculator
        E = test_structure.get_potential_energy()
        assert np.isfinite(E)

    def test_energy_reasonable(self, library_calculator, test_structure):
        """Test energy is in reasonable range (not NaN or extremely large)."""
        test_structure.calc = library_calculator
        E = test_structure.get_potential_energy()
        E_per_atom = E / len(test_structure)
        # Energy scale depends on model reference - just check it's not absurdly large
        assert abs(E_per_atom) < 10000, f"Energy per atom {E_per_atom} seems unreasonable"

    def test_forces_finite(self, library_calculator, test_structure):
        """Test that forces are finite."""
        test_structure.calc = library_calculator
        F = test_structure.get_forces()
        assert np.all(np.isfinite(F))
        assert F.shape == (len(test_structure), 3)

    def test_forces_perfect_crystal(self, library_calculator, test_structure):
        """Test perfect crystal has near-zero forces."""
        test_structure.calc = library_calculator
        F = test_structure.get_forces()
        max_force = np.abs(F).max()
        assert max_force < 0.5, f"Max force {max_force} too large for perfect crystal"

    def test_forces_perturbed(self, library_calculator, perturbed_structure):
        """Test perturbed structure has non-zero forces."""
        perturbed_structure.calc = library_calculator
        F = perturbed_structure.get_forces()
        max_force = np.abs(F).max()
        assert max_force > 1e-4, "Perturbed structure should have non-zero forces"

    def test_stress_finite(self, library_calculator, test_structure):
        """Test that stress is finite."""
        test_structure.calc = library_calculator
        S = test_structure.get_stress()
        assert np.all(np.isfinite(S))
        assert len(S) == 6  # Voigt notation

    def test_energy_scales_with_size(self, library_calculator, test_structure, test_supercell):
        """Test energy per atom is consistent across system sizes."""
        test_structure.calc = library_calculator
        test_supercell.calc = library_calculator

        E_unit = test_structure.get_potential_energy()
        E_super = test_supercell.get_potential_energy()

        E_per_atom_unit = E_unit / len(test_structure)
        E_per_atom_super = E_super / len(test_supercell)

        rel_diff = abs(E_per_atom_unit - E_per_atom_super) / max(abs(E_per_atom_unit), 1e-10)
        assert rel_diff < 0.05, f"Energy per atom differs by {rel_diff*100:.1f}%"


@pytest.mark.requires_library
class TestLibraryForceConsistency:
    """Test force consistency with finite differences."""

    def test_finite_difference_forces(self, library_calculator, perturbed_structure):
        """Test analytic forces match finite difference."""
        perturbed_structure.calc = library_calculator

        # Get analytic forces
        F_analytic = perturbed_structure.get_forces()

        # Compute finite difference for first atom, x-direction
        h = 1e-5
        pos = perturbed_structure.positions.copy()

        pos[0, 0] += h
        perturbed_structure.positions = pos
        E_p = perturbed_structure.get_potential_energy()

        pos[0, 0] -= 2*h
        perturbed_structure.positions = pos
        E_m = perturbed_structure.get_potential_energy()

        F_fd_x = -(E_p - E_m) / (2*h)

        # Restore
        pos[0, 0] += h
        perturbed_structure.positions = pos

        # Compare (FD accuracy is limited)
        err = abs(F_analytic[0, 0] - F_fd_x)
        assert err < 0.01, f"Force FD error {err} too large"


@pytest.mark.requires_library
class TestLibraryReusability:
    """Test calculator can be reused for multiple calculations."""

    def test_multiple_structures(self, library_calculator, test_structure, test_supercell, perturbed_structure):
        """Test same calculator works on multiple structures."""
        results = []

        for atoms in [test_structure, test_supercell, perturbed_structure]:
            atoms.calc = library_calculator
            E = atoms.get_potential_energy()
            F = atoms.get_forces()
            results.append((E, F))

        # All results should be finite
        for E, F in results:
            assert np.isfinite(E)
            assert np.all(np.isfinite(F))

    def test_repeated_calculations(self, library_calculator, test_structure):
        """Test repeated calculations give same result."""
        test_structure.calc = library_calculator

        E1 = test_structure.get_potential_energy()
        F1 = test_structure.get_forces().copy()

        E2 = test_structure.get_potential_energy()
        F2 = test_structure.get_forces().copy()

        assert E1 == E2
        assert np.allclose(F1, F2)
