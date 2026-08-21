"""
Unit tests for ACECalculator.

Run with:
    pytest -v tests/test_calculator.py

For tests requiring Julia:
    pytest -v tests/test_calculator.py -m requires_julia
"""

import pytest
import numpy as np
from pathlib import Path


class TestImports:
    """Test that the package can be imported."""

    def test_import_package(self):
        """Test importing the main package."""
        import ase_ace
        assert hasattr(ase_ace, 'ACECalculator')
        assert hasattr(ase_ace, '__version__')

    def test_import_calculator(self):
        """Test importing ACECalculator directly."""
        from ase_ace import ACECalculator
        assert ACECalculator is not None

    def test_import_server(self):
        """Test importing server module."""
        from ase_ace.server import JuliaACEServer, find_free_port
        assert JuliaACEServer is not None
        assert callable(find_free_port)

    def test_import_utils(self):
        """Test importing utils module."""
        from ase_ace.utils import find_julia, check_julia_version
        assert callable(find_julia)
        assert callable(check_julia_version)


class TestUtilities:
    """Test utility functions."""

    def test_find_free_port(self):
        """Test that find_free_port returns a valid port."""
        from ase_ace.server import find_free_port

        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 < port < 65536

    def test_find_free_port_unique(self):
        """Test that find_free_port returns different ports."""
        from ase_ace.server import find_free_port

        ports = [find_free_port() for _ in range(5)]
        # Ports might occasionally collide, but 5 in a row is unlikely
        assert len(set(ports)) >= 3

    def test_find_julia(self, julia_available):
        """Test finding Julia executable."""
        from ase_ace.utils import find_julia

        result = find_julia()
        if julia_available:
            assert result is not None
            assert 'julia' in result.lower()
        else:
            # May or may not be None depending on system

            pass

    @pytest.mark.requires_julia
    def test_check_julia_version(self, julia_available):
        """Test checking Julia version."""
        if not julia_available:
            pytest.skip("Julia not available")

        from ase_ace.utils import check_julia_version

        major, minor, patch = check_julia_version()
        assert major >= 1
        assert minor >= 0
        assert patch >= 0


class TestCalculatorInit:
    """Test calculator initialization (without starting Julia)."""

    def test_init_missing_model(self):
        """Test that missing model raises FileNotFoundError."""
        from ase_ace import ACECalculator

        with pytest.raises(FileNotFoundError):
            ACECalculator('/nonexistent/model.json')

    def test_init_properties(self, test_model_path, julia_available):
        """Test calculator properties after init."""
        if not julia_available:
            pytest.skip("Julia not available")

        from ase_ace import ACECalculator

        calc = ACECalculator(
            test_model_path,
            num_threads=2,
            port=0,
        )

        assert calc.model_path.exists()
        assert calc.num_threads == 2
        assert calc.port == 0
        assert calc._started is False

        # Don't call calc.close() since we never started


@pytest.mark.requires_julia
class TestCalculatorCalculations:
    """Test actual calculations with Julia."""

    def test_energy_finite(self, ace_calculator, si_diamond):
        """Test that energy is finite."""
        si_diamond.calc = ace_calculator
        E = si_diamond.get_potential_energy()
        assert np.isfinite(E)

    def test_energy_reasonable(self, ace_calculator, si_diamond):
        """Test energy is in reasonable range."""
        si_diamond.calc = ace_calculator
        E = si_diamond.get_potential_energy()
        E_per_atom = E / len(si_diamond)
        # Energy per atom should be reasonable (not NaN or huge)
        assert abs(E_per_atom) < 100

    def test_forces_finite(self, ace_calculator, si_diamond):
        """Test that forces are finite."""
        si_diamond.calc = ace_calculator
        F = si_diamond.get_forces()
        assert np.all(np.isfinite(F))
        assert F.shape == (len(si_diamond), 3)

    def test_forces_perfect_crystal(self, ace_calculator, si_diamond):
        """Test perfect crystal has near-zero forces."""
        si_diamond.calc = ace_calculator
        F = si_diamond.get_forces()
        max_force = np.abs(F).max()
        # Forces should be very small for equilibrium structure
        assert max_force < 0.1, f"Max force {max_force} too large for perfect crystal"

    def test_forces_perturbed(self, ace_calculator, perturbed_si):
        """Test perturbed structure has non-zero forces."""
        perturbed_si.calc = ace_calculator
        F = perturbed_si.get_forces()
        max_force = np.abs(F).max()
        assert max_force > 1e-4, "Perturbed structure should have non-zero forces"

    def test_stress_finite(self, ace_calculator, si_diamond):
        """Test that stress is finite."""
        si_diamond.calc = ace_calculator
        S = si_diamond.get_stress()
        assert np.all(np.isfinite(S))
        assert len(S) == 6  # Voigt notation

    def test_energy_scales_with_size(self, ace_calculator, si_diamond, si_supercell):
        """Test energy per atom is consistent across system sizes."""
        si_diamond.calc = ace_calculator
        si_supercell.calc = ace_calculator

        E_unit = si_diamond.get_potential_energy()
        E_super = si_supercell.get_potential_energy()

        E_per_atom_unit = E_unit / len(si_diamond)
        E_per_atom_super = E_super / len(si_supercell)

        # Should be within 5% (small systems may have boundary effects)
        rel_diff = abs(E_per_atom_unit - E_per_atom_super) / max(abs(E_per_atom_unit), 1e-10)
        assert rel_diff < 0.05, f"Energy per atom differs by {rel_diff*100:.1f}%"


@pytest.mark.requires_julia
class TestCalculatorLifecycle:
    """Test calculator lifecycle management."""

    def test_context_manager(self, test_model_path, julia_project_path):
        """Test using calculator as context manager."""
        from ase_ace import ACECalculator
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)

        with ACECalculator(test_model_path, num_threads=1,
                          julia_project=julia_project_path,
                          timeout=120.0) as calc:
            atoms.calc = calc
            E = atoms.get_potential_energy()
            assert np.isfinite(E)

        # After context manager, calculator should be closed
        assert calc._started is False

    def test_manual_close(self, test_model_path, julia_project_path):
        """Test manual close of calculator."""
        from ase_ace import ACECalculator
        from ase.build import bulk

        calc = ACECalculator(test_model_path, num_threads=1,
                            julia_project=julia_project_path,
                            timeout=120.0)
        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.calc = calc

        E = atoms.get_potential_energy()
        assert np.isfinite(E)

        calc.close()
        assert calc._started is False

    def test_multiple_calculations(self, ace_calculator, si_diamond):
        """Test multiple calculations with same calculator."""
        si_diamond.calc = ace_calculator

        E1 = si_diamond.get_potential_energy()
        F1 = si_diamond.get_forces()

        # Perturb and calculate again
        si_diamond.positions[0, 0] += 0.01
        E2 = si_diamond.get_potential_energy()
        F2 = si_diamond.get_forces()

        # Energies should be different
        assert E1 != E2
        assert not np.allclose(F1, F2)


@pytest.mark.requires_julia
class TestForceConsistency:
    """Test force consistency with finite differences."""

    def test_finite_difference_forces(self, ace_calculator, perturbed_si):
        """Test analytic forces match finite difference."""
        perturbed_si.calc = ace_calculator

        # Get analytic forces
        F_analytic = perturbed_si.get_forces()

        # Compute finite difference for first atom, x-direction
        h = 1e-5
        pos = perturbed_si.positions.copy()

        pos[0, 0] += h
        perturbed_si.positions = pos
        E_p = perturbed_si.get_potential_energy()

        pos[0, 0] -= 2*h
        perturbed_si.positions = pos
        E_m = perturbed_si.get_potential_energy()

        F_fd_x = -(E_p - E_m) / (2*h)

        # Restore
        pos[0, 0] += h
        perturbed_si.positions = pos

        # Compare (FD accuracy is limited)
        err = abs(F_analytic[0, 0] - F_fd_x)
        assert err < 0.01, f"Force FD error {err} too large"
