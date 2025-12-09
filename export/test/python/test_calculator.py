"""
Comprehensive tests for ACE Python/ASE calculator.

Run with:
    pytest -v test_calculator.py

Environment:
    ACE_LIB_PATH=/path/to/libace.so
    LD_LIBRARY_PATH=/path/to/julia/lib
"""

import pytest
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units


class TestLibraryLoading:
    """Test basic library loading and info functions."""

    def test_calculator_creation(self, ace_calculator):
        """Test that calculator can be created."""
        assert ace_calculator is not None

    def test_cutoff(self, ace_calculator):
        """Test cutoff is reasonable."""
        assert ace_calculator.cutoff > 0
        assert ace_calculator.cutoff < 10

    def test_species(self, ace_calculator):
        """Test species info."""
        assert 14 in ace_calculator.species  # Si


class TestEnergyCalculation:
    """Test energy calculations."""

    def test_energy_finite(self, ace_calculator, si_diamond):
        """Test energy is finite."""
        si_diamond.calc = ace_calculator
        E = si_diamond.get_potential_energy()
        assert np.isfinite(E)

    def test_energy_reasonable(self, ace_calculator, si_diamond):
        """Test energy is reasonable (within expected range for Si)."""
        si_diamond.calc = ace_calculator
        E = si_diamond.get_potential_energy()
        E_per_atom = E / len(si_diamond)
        # Test model may not have negative cohesive energy (depends on reference)
        # Just check it's in a reasonable range for a test potential
        assert abs(E_per_atom) < 100  # Not wildly wrong

    def test_energy_per_atom(self, ace_calculator, si_diamond, si_supercell):
        """Test energy scales with system size."""
        si_diamond.calc = ace_calculator
        si_supercell.calc = ace_calculator

        E_unit = si_diamond.get_potential_energy()
        E_super = si_supercell.get_potential_energy()

        # 2x2x2 supercell has 8x atoms
        E_per_atom_unit = E_unit / len(si_diamond)
        E_per_atom_super = E_super / len(si_supercell)

        # Should be within 1% (boundary effects)
        rel_diff = abs(E_per_atom_unit - E_per_atom_super) / abs(E_per_atom_unit)
        assert rel_diff < 0.01


class TestForceCalculation:
    """Test force calculations."""

    def test_forces_finite(self, ace_calculator, si_diamond):
        """Test forces are finite."""
        si_diamond.calc = ace_calculator
        F = si_diamond.get_forces()
        assert np.all(np.isfinite(F))

    def test_forces_perfect_crystal(self, ace_calculator, si_diamond):
        """Test perfect crystal has near-zero forces."""
        si_diamond.calc = ace_calculator
        F = si_diamond.get_forces()
        max_force = np.abs(F).max()
        assert max_force < 1e-6, f"Max force {max_force} too large for perfect crystal"

    def test_forces_perturbed(self, ace_calculator, perturbed_si):
        """Test perturbed structure has non-zero forces."""
        perturbed_si.calc = ace_calculator
        F = perturbed_si.get_forces()
        max_force = np.abs(F).max()
        assert max_force > 1e-4, "Perturbed structure should have non-zero forces"

    def test_forces_finite_difference(self, ace_calculator, perturbed_si):
        """Test analytic forces match finite difference."""
        perturbed_si.calc = ace_calculator

        # Get analytic forces
        F_analytic = perturbed_si.get_forces()

        # Compute finite difference forces for first atom
        h = 1e-6
        F_fd = np.zeros(3)

        for alpha in range(3):
            pos = perturbed_si.positions.copy()

            pos[0, alpha] += h
            perturbed_si.positions = pos
            E_p = perturbed_si.get_potential_energy()

            pos[0, alpha] -= 2*h
            perturbed_si.positions = pos
            E_m = perturbed_si.get_potential_energy()

            F_fd[alpha] = -(E_p - E_m) / (2*h)

            # Restore
            pos[0, alpha] += h
            perturbed_si.positions = pos

        max_err = np.max(np.abs(F_analytic[0] - F_fd))
        # FD with h=1e-6 typically gives ~1e-4 to 1e-3 accuracy
        assert max_err < 1e-2, f"Force finite difference error {max_err} too large"


class TestStressCalculation:
    """Test stress tensor calculations."""

    def test_stress_finite(self, ace_calculator, si_diamond):
        """Test stress is finite."""
        si_diamond.calc = ace_calculator
        S = si_diamond.get_stress()
        assert np.all(np.isfinite(S))

    def test_stress_voigt(self, ace_calculator, si_diamond):
        """Test stress is in Voigt notation (6 components)."""
        si_diamond.calc = ace_calculator
        S = si_diamond.get_stress()
        assert len(S) == 6

    def test_stress_cubic_symmetry(self, ace_calculator, si_diamond):
        """Test cubic crystal has isotropic stress."""
        si_diamond.calc = ace_calculator
        S = si_diamond.get_stress()

        # xx, yy, zz should be equal for cubic
        diag = S[:3]
        rel_diff = np.std(diag) / np.abs(np.mean(diag))
        assert rel_diff < 0.01, "Cubic crystal should have isotropic stress"


class TestGeometryOptimization:
    """Test geometry optimization."""

    def test_bfgs_convergence(self, ace_calculator, perturbed_si):
        """Test BFGS optimization converges."""
        perturbed_si.calc = ace_calculator

        initial_max_force = np.abs(perturbed_si.get_forces()).max()

        opt = BFGS(perturbed_si, logfile=None)
        converged = opt.run(fmax=0.01, steps=50)

        final_max_force = np.abs(perturbed_si.get_forces()).max()

        assert final_max_force < initial_max_force
        assert final_max_force < 0.01 or opt.nsteps < 50


class TestMolecularDynamics:
    """Test molecular dynamics."""

    def test_nve_energy_conservation(self, ace_calculator, si_supercell):
        """Test NVE energy conservation."""
        si_supercell.calc = ace_calculator

        # Initialize velocities at 100 K
        MaxwellBoltzmannDistribution(si_supercell, temperature_K=100)
        momenta = si_supercell.get_momenta()
        momenta -= momenta.mean(axis=0)
        si_supercell.set_momenta(momenta)

        # Run NVE
        dyn = VelocityVerlet(si_supercell, timestep=0.5 * units.fs)

        energies = []
        for _ in range(20):
            dyn.run(5)
            e_total = si_supercell.get_kinetic_energy() + si_supercell.get_potential_energy()
            energies.append(e_total)

        energies = np.array(energies)
        drift = np.abs(energies[-1] - energies[0])
        std = np.std(energies)

        # Energy conservation check
        assert drift < 1e-2, f"Energy drift {drift} eV too large"
        assert std < 1e-2, f"Energy fluctuation {std} eV too large"


class TestConsistency:
    """Test consistency between different calculation modes."""

    def test_energy_forces_consistency(self, ace_calculator, perturbed_si):
        """Test that get_potential_energy() and get_forces() are consistent."""
        perturbed_si.calc = ace_calculator

        # Call separately
        E1 = perturbed_si.get_potential_energy()
        F1 = perturbed_si.get_forces()

        # Force recalculation
        perturbed_si.calc.reset()

        # Call in reverse order
        F2 = perturbed_si.get_forces()
        E2 = perturbed_si.get_potential_energy()

        assert abs(E1 - E2) < 1e-10
        assert np.allclose(F1, F2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
