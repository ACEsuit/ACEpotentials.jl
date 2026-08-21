"""
Tests for ACE descriptor computation.

These tests verify that the get_descriptors() method returns correct
basis vectors that can be used for fitting, analysis, and transfer learning.
"""

import os
import pytest
import numpy as np
from ase.build import bulk


class TestDescriptors:
    """Tests for descriptor computation."""

    @pytest.fixture
    def ace_calculator(self):
        """Get ACE calculator with compiled library."""
        lib_path = os.environ.get('ACE_TEST_LIBRARY')
        if not lib_path or not os.path.exists(lib_path):
            pytest.skip("ACE_TEST_LIBRARY not set or library not found")

        from ase_ace import ACELibraryCalculator
        return ACELibraryCalculator(lib_path)

    def test_n_basis_property(self, ace_calculator):
        """Test that n_basis property returns a positive integer."""
        n_basis = ace_calculator.n_basis
        assert isinstance(n_basis, int)
        assert n_basis > 0
        print(f"Number of basis functions: {n_basis}")

    def test_descriptor_shape(self, ace_calculator):
        """Test that descriptors have correct shape (natoms, n_basis)."""
        atoms = bulk('Si', 'diamond', a=5.43)
        D = ace_calculator.get_descriptors(atoms)

        assert D.shape == (len(atoms), ace_calculator.n_basis)
        print(f"Descriptor shape: {D.shape}")

    def test_descriptor_finite(self, ace_calculator):
        """Test that all descriptor values are finite."""
        atoms = bulk('Si', 'diamond', a=5.43)
        D = ace_calculator.get_descriptors(atoms)

        assert np.all(np.isfinite(D)), "Descriptors contain non-finite values"

    def test_descriptor_nonzero(self, ace_calculator):
        """Test that descriptors are not all zero."""
        atoms = bulk('Si', 'diamond', a=5.43)
        D = ace_calculator.get_descriptors(atoms)

        assert np.any(D != 0), "All descriptors are zero"

    def test_descriptor_deterministic(self, ace_calculator):
        """Test that descriptors are deterministic."""
        atoms = bulk('Si', 'diamond', a=5.43)

        D1 = ace_calculator.get_descriptors(atoms)
        D2 = ace_calculator.get_descriptors(atoms)

        np.testing.assert_array_equal(D1, D2, "Descriptors not deterministic")

    def test_descriptor_supercell_scaling(self, ace_calculator):
        """Test descriptor scaling with supercell."""
        # Unit cell
        atoms_1x1 = bulk('Si', 'diamond', a=5.43)
        D_1x1 = ace_calculator.get_descriptors(atoms_1x1)

        # 2x2x2 supercell
        atoms_2x2 = atoms_1x1 * (2, 2, 2)
        D_2x2 = ace_calculator.get_descriptors(atoms_2x2)

        # Each atom in the supercell should have the same descriptor
        # as the corresponding atom in the unit cell (for perfect crystal)
        assert D_2x2.shape[0] == 8 * D_1x1.shape[0]
        assert D_2x2.shape[1] == D_1x1.shape[1]

        # All atoms in perfect crystal should have same descriptor
        # (due to translation symmetry)
        for i in range(1, len(atoms_2x2)):
            np.testing.assert_allclose(
                D_2x2[i], D_2x2[0], rtol=1e-10,
                err_msg=f"Atom {i} has different descriptor than atom 0"
            )

    def test_perturbed_descriptors_differ(self, ace_calculator):
        """Test that perturbing atoms changes descriptors."""
        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)

        D_orig = ace_calculator.get_descriptors(atoms)

        # Perturb one atom
        atoms_perturbed = atoms.copy()
        atoms_perturbed.positions[0] += [0.1, 0.0, 0.0]

        D_pert = ace_calculator.get_descriptors(atoms_perturbed)

        # Descriptors should be different
        assert not np.allclose(D_orig, D_pert), \
            "Descriptors unchanged after perturbation"

        # The perturbed atom and its neighbors should have different descriptors
        assert not np.allclose(D_orig[0], D_pert[0]), \
            "Perturbed atom descriptor unchanged"

    def test_isolated_atom_zero_descriptor(self, ace_calculator):
        """Test that isolated atom has zero descriptor (no neighbors)."""
        from ase import Atoms

        # Create isolated atom with large cell (no neighbors within cutoff)
        cell_size = 2 * ace_calculator.cutoff + 10.0
        atoms = Atoms('Si', positions=[[0, 0, 0]],
                      cell=[cell_size, cell_size, cell_size],
                      pbc=True)

        D = ace_calculator.get_descriptors(atoms)

        assert D.shape == (1, ace_calculator.n_basis)
        np.testing.assert_array_equal(
            D[0], np.zeros(ace_calculator.n_basis),
            "Isolated atom should have zero descriptor"
        )


class TestDescriptorEnergyRelation:
    """Tests verifying E = sum(descriptors @ weights) for linear models."""

    @pytest.fixture
    def ace_calculator(self):
        """Get ACE calculator with compiled library."""
        lib_path = os.environ.get('ACE_TEST_LIBRARY')
        if not lib_path or not os.path.exists(lib_path):
            pytest.skip("ACE_TEST_LIBRARY not set or library not found")

        from ase_ace import ACELibraryCalculator
        return ACELibraryCalculator(lib_path)

    def test_energy_descriptor_consistency(self, ace_calculator):
        """
        Test that energy and descriptors are consistent.

        For a perfect crystal, if all atoms have the same descriptor D,
        then E_total = n_atoms * E_per_atom where E_per_atom = D @ weights + E0.

        While we can't directly access weights, we can verify that:
        - E_per_atom is consistent across different supercell sizes
        - Descriptors scale correctly with system size
        """
        # Compare unit cell and supercell
        atoms_1 = bulk('Si', 'diamond', a=5.43)
        atoms_1.calc = ace_calculator

        atoms_8 = atoms_1 * (2, 2, 2)
        atoms_8.calc = ace_calculator

        E_1 = atoms_1.get_potential_energy()
        E_8 = atoms_8.get_potential_energy()

        D_1 = ace_calculator.get_descriptors(atoms_1)
        D_8 = ace_calculator.get_descriptors(atoms_8)

        # Energy per atom should be the same
        E_per_atom_1 = E_1 / len(atoms_1)
        E_per_atom_8 = E_8 / len(atoms_8)

        np.testing.assert_allclose(
            E_per_atom_1, E_per_atom_8, rtol=1e-10,
            err_msg="Energy per atom differs between unit cell and supercell"
        )

        # Total descriptor should scale with natoms
        D_sum_1 = np.sum(D_1, axis=0)
        D_sum_8 = np.sum(D_8, axis=0)

        # For perfect crystal: D_sum_8 = 8 * D_sum_1
        np.testing.assert_allclose(
            D_sum_8, 8 * D_sum_1, rtol=1e-10,
            err_msg="Total descriptors don't scale with system size"
        )
