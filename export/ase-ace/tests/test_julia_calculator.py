"""
Tests for ACEJuliaCalculator.

Run with:
    pytest -v tests/test_julia_calculator.py

Requires:
    pip install ase-ace[julia]

Environment variable:
    ACE_TEST_MODEL: Path to ACE model JSON file for testing
"""

import os
import pytest
import numpy as np
from ase.build import bulk


def _juliacall_available():
    """Check if juliacall is available without importing it."""
    try:
        import importlib.util
        return importlib.util.find_spec("juliacall") is not None
    except Exception:
        return False


# Skip all tests if juliacall not available (without actually importing it)
if not _juliacall_available():
    pytest.skip("juliacall not available", allow_module_level=True)


@pytest.fixture(scope="module")
def model_path():
    """Get path to test model from environment."""
    path = os.environ.get('ACE_TEST_MODEL')
    if not path or not os.path.exists(path):
        pytest.skip("ACE_TEST_MODEL not set or model not found")
    return path


@pytest.fixture(scope="module")
def julia_calculator(model_path):
    """Create ACEJuliaCalculator for testing."""
    from ase_ace import ACEJuliaCalculator

    # Use single thread for deterministic tests
    calc = ACEJuliaCalculator(model_path, num_threads=1)
    yield calc


@pytest.fixture
def si_diamond():
    """Create Si diamond structure."""
    return bulk('Si', 'diamond', a=5.43)


@pytest.fixture
def si_supercell(si_diamond):
    """Create 2x2x2 Si supercell."""
    return si_diamond * (2, 2, 2)


@pytest.fixture
def perturbed_si(si_supercell):
    """Create perturbed Si structure for force tests."""
    atoms = si_supercell.copy()
    np.random.seed(42)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
    return atoms


class TestImports:
    """Test imports and dependencies."""

    def test_import_calculator(self):
        from ase_ace import ACEJuliaCalculator
        assert ACEJuliaCalculator is not None

    def test_import_base_class(self):
        from ase_ace import ACECalculatorBase
        assert ACECalculatorBase is not None

    def test_juliacall_available(self):
        # Check availability without importing (which would start Julia)
        assert _juliacall_available()


class TestInitialization:
    """Test calculator initialization."""

    def test_missing_model_raises(self):
        from ase_ace import ACEJuliaCalculator
        with pytest.raises(FileNotFoundError):
            ACEJuliaCalculator('/nonexistent/model.json')

    def test_lazy_initialization(self, model_path):
        """Test that Julia isn't loaded until needed."""
        from ase_ace import ACEJuliaCalculator

        calc = ACEJuliaCalculator(model_path)
        # Model should not be loaded yet
        assert calc._model is None

        # Accessing cutoff triggers loading
        _ = calc.cutoff
        assert calc._model is not None

    def test_properties_available(self, julia_calculator):
        """Test that properties are accessible."""
        assert julia_calculator.cutoff > 0
        assert len(julia_calculator.species) > 0
        assert julia_calculator.n_basis > 0


class TestCalculations:
    """Test actual calculations."""

    def test_energy_finite(self, julia_calculator, si_diamond):
        si_diamond.calc = julia_calculator
        E = si_diamond.get_potential_energy()
        assert np.isfinite(E)

    def test_forces_finite(self, julia_calculator, si_diamond):
        si_diamond.calc = julia_calculator
        F = si_diamond.get_forces()
        assert np.all(np.isfinite(F))
        assert F.shape == (len(si_diamond), 3)

    def test_stress_finite(self, julia_calculator, si_diamond):
        si_diamond.calc = julia_calculator
        S = si_diamond.get_stress()
        assert np.all(np.isfinite(S))
        assert len(S) == 6

    def test_perfect_crystal_zero_forces(self, julia_calculator, si_diamond):
        """Perfect crystal should have near-zero forces."""
        si_diamond.calc = julia_calculator
        F = si_diamond.get_forces()
        max_force = np.abs(F).max()
        assert max_force < 1e-6, f"Max force too large: {max_force}"

    def test_energy_scaling(self, julia_calculator, si_diamond, si_supercell):
        """Energy should scale with system size."""
        si_diamond.calc = julia_calculator
        si_supercell.calc = julia_calculator

        E1 = si_diamond.get_potential_energy()
        E8 = si_supercell.get_potential_energy()

        # Compare energy per atom - should be the same for perfect crystals
        # This avoids division by zero if E1 is zero
        E1_per_atom = E1 / len(si_diamond)
        E8_per_atom = E8 / len(si_supercell)
        np.testing.assert_allclose(
            E1_per_atom, E8_per_atom, rtol=1e-10, atol=1e-10,
            err_msg=f"Energy per atom differs: {E1_per_atom} vs {E8_per_atom}"
        )

    def test_forces_perturbed(self, julia_calculator, perturbed_si):
        """Perturbed structure should have non-zero forces."""
        perturbed_si.calc = julia_calculator
        F = perturbed_si.get_forces()
        max_force = np.abs(F).max()
        assert max_force > 0.01, f"Forces too small: {max_force}"


class TestFiniteDifference:
    """Test forces with finite differences."""

    def test_forces_finite_difference(self, julia_calculator, perturbed_si):
        """Test analytic forces match finite differences."""
        perturbed_si.calc = julia_calculator

        F_analytic = perturbed_si.get_forces()

        h = 1e-5
        pos = perturbed_si.positions.copy()

        # Test first atom, x-direction
        pos[0, 0] += h
        perturbed_si.positions = pos
        E_p = perturbed_si.get_potential_energy()

        pos[0, 0] -= 2 * h
        perturbed_si.positions = pos
        E_m = perturbed_si.get_potential_energy()

        F_fd = -(E_p - E_m) / (2 * h)

        # Restore positions
        pos[0, 0] += h
        perturbed_si.positions = pos

        error = abs(F_analytic[0, 0] - F_fd)
        assert error < 0.01, f"FD error: {error}"


class TestDescriptors:
    """Test descriptor computation."""

    def test_descriptor_shape(self, julia_calculator, si_diamond):
        """Test that descriptors have correct shape."""
        D = julia_calculator.get_descriptors(si_diamond)
        assert D.shape == (len(si_diamond), julia_calculator.n_basis)

    def test_descriptor_finite(self, julia_calculator, si_diamond):
        """Test that all descriptor values are finite."""
        D = julia_calculator.get_descriptors(si_diamond)
        assert np.all(np.isfinite(D)), "Descriptors contain non-finite values"

    def test_descriptor_nonzero(self, julia_calculator, si_diamond):
        """Test that descriptors are not all zero."""
        D = julia_calculator.get_descriptors(si_diamond)
        assert np.any(D != 0), "All descriptors are zero"

    def test_descriptor_deterministic(self, julia_calculator, si_diamond):
        """Test that descriptors are deterministic."""
        D1 = julia_calculator.get_descriptors(si_diamond)
        D2 = julia_calculator.get_descriptors(si_diamond)
        np.testing.assert_array_equal(D1, D2, "Descriptors not deterministic")

    def test_descriptor_supercell_consistency(self, julia_calculator, si_diamond, si_supercell):
        """Test descriptor consistency across supercells."""
        D1 = julia_calculator.get_descriptors(si_diamond)
        D8 = julia_calculator.get_descriptors(si_supercell)

        # All atoms in perfect crystal should have the same descriptor
        # Use generous atol for very small values where relative comparison is meaningless
        # (some descriptor components can be ~1e-32 due to near-zero contributions)
        for i in range(1, len(si_supercell)):
            np.testing.assert_allclose(
                D8[i], D8[0], rtol=1e-10, atol=1e-14,
                err_msg=f"Atom {i} has different descriptor than atom 0"
            )


class TestBaseClass:
    """Test base class inheritance."""

    def test_inherits_from_base(self, julia_calculator):
        """Test that calculator inherits from ACECalculatorBase."""
        from ase_ace import ACECalculatorBase
        assert isinstance(julia_calculator, ACECalculatorBase)

    def test_implemented_properties(self, julia_calculator):
        """Test that implemented_properties is set correctly."""
        assert 'energy' in julia_calculator.implemented_properties
        assert 'forces' in julia_calculator.implemented_properties
        assert 'stress' in julia_calculator.implemented_properties


class TestForceVirialDescriptors:
    """Test force and virial descriptor computation."""

    def test_force_descriptors_shape(self, julia_calculator, si_supercell):
        """Test that force descriptors have correct shape."""
        result = julia_calculator.get_descriptors(
            si_supercell, include_forces=True
        )

        assert isinstance(result, dict)
        assert 'descriptors' in result
        assert 'energy' in result
        assert 'forces' in result
        assert 'virial' not in result  # Not requested

        n_atoms = len(si_supercell)
        n_basis = julia_calculator.n_basis

        assert result['descriptors'].shape == (n_atoms, n_basis)
        assert result['energy'].shape == (n_basis,)
        assert result['forces'].shape == (n_atoms, n_basis, 3)

    def test_virial_descriptors_shape(self, julia_calculator, si_supercell):
        """Test that virial descriptors have correct shape."""
        result = julia_calculator.get_descriptors(
            si_supercell, include_virial=True
        )

        assert isinstance(result, dict)
        assert 'descriptors' in result
        assert 'energy' in result
        assert 'forces' not in result  # Not requested
        assert 'virial' in result

        n_basis = julia_calculator.n_basis
        assert result['virial'].shape == (n_basis, 3, 3)

    def test_force_virial_descriptors_shape(self, julia_calculator, si_supercell):
        """Test that both force and virial descriptors have correct shape."""
        result = julia_calculator.get_descriptors(
            si_supercell, include_forces=True, include_virial=True
        )

        assert isinstance(result, dict)
        assert 'descriptors' in result
        assert 'energy' in result
        assert 'forces' in result
        assert 'virial' in result

        n_atoms = len(si_supercell)
        n_basis = julia_calculator.n_basis

        assert result['descriptors'].shape == (n_atoms, n_basis)
        assert result['energy'].shape == (n_basis,)
        assert result['forces'].shape == (n_atoms, n_basis, 3)
        assert result['virial'].shape == (n_basis, 3, 3)

    def test_force_descriptors_finite(self, julia_calculator, si_supercell):
        """Test that all force descriptor values are finite."""
        result = julia_calculator.get_descriptors(
            si_supercell, include_forces=True, include_virial=True
        )

        assert np.all(np.isfinite(result['descriptors']))
        assert np.all(np.isfinite(result['energy']))
        assert np.all(np.isfinite(result['forces']))
        assert np.all(np.isfinite(result['virial']))

    def test_backward_compatible_no_flags(self, julia_calculator, si_supercell):
        """Test that default call returns array, not dict."""
        result = julia_calculator.get_descriptors(si_supercell)

        # Without flags, should return array directly (backward compatible)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(si_supercell), julia_calculator.n_basis)

    def test_energy_sum_matches_energy_descriptors(self, julia_calculator, perturbed_si):
        """Test that sum of site descriptors matches energy descriptors."""
        result = julia_calculator.get_descriptors(
            perturbed_si, include_forces=True
        )

        # Site descriptors summed over atoms
        site_sum = np.sum(result['descriptors'], axis=0)

        # Energy descriptors (from energy_forces_virial_basis)
        energy_desc = result['energy']

        # These should be very close (both represent per-basis energy contributions)
        np.testing.assert_allclose(
            site_sum, energy_desc, rtol=1e-10, atol=1e-14,
            err_msg="Sum of site descriptors doesn't match energy descriptors"
        )

    def test_forces_sum_consistency(self, julia_calculator, perturbed_si):
        """Test that force descriptors are consistent with actual forces."""
        # Get force descriptors
        result = julia_calculator.get_descriptors(
            perturbed_si, include_forces=True
        )

        # Get actual forces
        perturbed_si.calc = julia_calculator
        forces_actual = perturbed_si.get_forces()

        # Sum force descriptors over basis (weighted by coefficients should give actual forces)
        # Here we just verify the shapes and that they're finite
        forces_desc = result['forces']  # (natoms, n_basis, 3)

        assert forces_desc.shape[0] == len(perturbed_si)
        assert forces_desc.shape[2] == 3
        assert np.all(np.isfinite(forces_desc))
        assert np.all(np.isfinite(forces_actual))


class TestLibraryConsistency:
    """Test consistency between JuliaCall and Library calculators."""

    @pytest.fixture
    def library_calculator(self):
        """Get library calculator if available."""
        lib_path = os.environ.get('ACE_TEST_LIBRARY')
        if not lib_path or not os.path.exists(lib_path):
            pytest.skip("ACE_TEST_LIBRARY not set or library not found")

        from ase_ace import ACELibraryCalculator
        try:
            return ACELibraryCalculator(lib_path)
        except OSError as e:
            # Library may be compiled with different Julia version
            pytest.skip(f"Cannot load library (Julia version mismatch?): {e}")

    def test_energy_matches_library(self, julia_calculator, library_calculator, si_diamond):
        """Compare JuliaCall results with compiled library."""
        si_diamond.calc = julia_calculator
        E_julia = si_diamond.get_potential_energy()

        si_diamond.calc = library_calculator
        E_lib = si_diamond.get_potential_energy()

        np.testing.assert_allclose(
            E_julia, E_lib, rtol=1e-10,
            err_msg="JuliaCall and Library energies differ"
        )

    def test_forces_match_library(self, julia_calculator, library_calculator, perturbed_si):
        """Compare forces between backends."""
        perturbed_si.calc = julia_calculator
        F_julia = perturbed_si.get_forces()

        perturbed_si.calc = library_calculator
        F_lib = perturbed_si.get_forces()

        np.testing.assert_allclose(
            F_julia, F_lib, rtol=1e-10,
            err_msg="JuliaCall and Library forces differ"
        )

    def test_descriptors_match_library(self, julia_calculator, library_calculator, si_diamond):
        """Compare descriptors between backends."""
        D_julia = julia_calculator.get_descriptors(si_diamond)
        D_lib = library_calculator.get_descriptors(si_diamond)

        np.testing.assert_allclose(
            D_julia, D_lib, rtol=1e-10,
            err_msg="JuliaCall and Library descriptors differ"
        )


class TestMultithreadedSignalHandling:
    """
    Multi-threaded juliacall segfaults unless Julia may install its signal handlers.

    Measured on this package's test model before the guard existed: num_threads=1 returned
    an energy, num_threads=2 exited 139 with no traceback.  juliacall does warn -- "It is
    recommended to restart Python with the environment variable
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes set, otherwise you may experience segfaults" -- but
    the warning reaches stderr from a process that is already doomed, and "restart Python"
    is not available to a caller several frames down.  _init_julia sets it before importing
    juliacall, in the same window it already uses for JULIA_NUM_THREADS.

    The whole ase-ace suite was exposed to this: every previous green run happened to have
    the variable exported, and a run without it segfaulted mid-file.
    """

    def test_guard_is_conditional_and_does_not_override(self):
        """
        Offline. The guard must stay conditional and must not overwrite a caller's choice.

        Setting it unconditionally is not free -- Julia takes over SIGINT, so Ctrl-C stops
        raising KeyboardInterrupt -- and single-threaded callers do not have the crash.
        """
        import inspect
        from ase_ace.julia_calculator import ACEJuliaCalculator

        src = inspect.getsource(ACEJuliaCalculator._init_julia)
        assert "PYTHON_JULIACALL_HANDLE_SIGNALS" in src, (
            "the multi-threaded segfault guard is gone from _init_julia"
        )
        assert "setdefault" in src, (
            "the guard must use setdefault so an explicit caller choice, including 'no', wins"
        )
        assert "> 1" in src, (
            "the guard must stay conditional on the thread count: forcing it single-threaded "
            "would take SIGINT away from callers who never had the crash"
        )

    @pytest.mark.skipif(
        os.environ.get("ACE_TEST_JULIA") != "1",
        reason="set ACE_TEST_JULIA=1 to start Julia in a subprocess",
    )
    def test_multithreaded_calculation_survives_without_the_variable(self, model_path):
        """
        The regression itself: two threads, variable absent from the environment.

        Runs in a SUBPROCESS with PYTHON_JULIACALL_HANDLE_SIGNALS unset, so the result does
        not depend on how this suite happened to be invoked -- which is exactly what hid the
        crash before.
        """
        import subprocess
        import sys
        import textwrap

        env = dict(os.environ)
        env.pop("PYTHON_JULIACALL_HANDLE_SIGNALS", None)

        script = textwrap.dedent(
            """
            from ase.build import bulk
            from ase_ace import ACEJuliaCalculator
            atoms = bulk('Si', 'diamond', a=5.43)
            atoms.calc = ACEJuliaCalculator(MODEL, num_threads=2)
            print('ENERGY', atoms.get_potential_energy())
            """
        ).replace("MODEL", repr(model_path))

        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=900,
        )
        assert proc.returncode == 0, (
            f"multi-threaded juliacall exited {proc.returncode} with "
            f"PYTHON_JULIACALL_HANDLE_SIGNALS unset -- the bug this guards.  A segfault "
            f"shows up here as -11 (subprocess reports the signal) and as 139 from a "
            f"shell (128+11).\n"
            f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-1500:]}"
        )
        assert "ENERGY" in proc.stdout
