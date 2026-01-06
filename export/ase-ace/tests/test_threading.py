"""
Threading tests for ACECalculator.

Tests that calculations are consistent across different thread counts.

Run with:
    pytest -v tests/test_threading.py -m requires_julia
"""

import pytest
import numpy as np
from pathlib import Path


# Reference values computed with 1 thread (filled in by first test run)
REFERENCE_ENERGY = None
REFERENCE_FORCES = None


@pytest.fixture(params=[1, 4, 8, 'auto'])
def num_threads(request):
    """Parametrized fixture for different thread counts."""
    return request.param


@pytest.fixture
def threaded_calculator(test_model_path, julia_available, julia_project_path, num_threads):
    """Create calculator with specified thread count."""
    if not julia_available:
        pytest.skip("Julia not available")

    from ase_ace import ACECalculator

    calc = ACECalculator(
        test_model_path,
        num_threads=num_threads,
        timeout=120.0,
        julia_project=julia_project_path,
    )
    yield calc
    calc.close()


@pytest.mark.requires_julia
class TestThreadingConsistency:
    """Test that results are consistent across thread counts."""

    def test_energy_consistent(self, threaded_calculator, si_supercell, num_threads):
        """Energy should be identical regardless of thread count."""
        global REFERENCE_ENERGY

        si_supercell.calc = threaded_calculator
        energy = si_supercell.get_potential_energy()

        assert np.isfinite(energy)

        if num_threads == 1:
            # First test with 1 thread sets reference
            REFERENCE_ENERGY = energy
        elif REFERENCE_ENERGY is not None:
            # Compare with reference
            assert np.isclose(energy, REFERENCE_ENERGY, rtol=1e-10), \
                f"Energy with {num_threads} threads ({energy}) differs from 1-thread ({REFERENCE_ENERGY})"

    def test_forces_consistent(self, threaded_calculator, si_supercell, num_threads):
        """Forces should be identical regardless of thread count."""
        global REFERENCE_FORCES

        si_supercell.calc = threaded_calculator
        forces = si_supercell.get_forces()

        assert np.all(np.isfinite(forces))

        if num_threads == 1:
            REFERENCE_FORCES = forces.copy()
        elif REFERENCE_FORCES is not None:
            assert np.allclose(forces, REFERENCE_FORCES, rtol=1e-10), \
                f"Forces with {num_threads} threads differ from 1-thread"


@pytest.mark.requires_julia
@pytest.mark.slow
class TestThreadingPerformance:
    """Test threading performance characteristics."""

    def test_larger_system_scaling(self, test_model_path, julia_available, julia_project_path):
        """
        Test that larger systems benefit from threading.

        This is a performance test, not a correctness test.
        We just verify it completes without checking speedup.
        """
        if not julia_available:
            pytest.skip("Julia not available")

        from ase_ace import ACECalculator
        from ase.build import bulk
        import time

        # Create a larger system
        atoms = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)  # 54 atoms

        times = {}
        for threads in [1, 4]:
            with ACECalculator(
                test_model_path,
                num_threads=threads,
                timeout=120.0,
                julia_project=julia_project_path,
            ) as calc:
                atoms.calc = calc

                # Warmup
                atoms.get_potential_energy()

                # Timed run
                start = time.perf_counter()
                for _ in range(3):
                    atoms.get_potential_energy()
                    atoms.get_forces()
                elapsed = time.perf_counter() - start

                times[threads] = elapsed

        # Just verify both completed
        assert 1 in times
        assert 4 in times

        # Log timing info (not a hard assertion)
        print(f"\nTiming (54 atoms, 3 iterations):")
        print(f"  1 thread:  {times[1]:.2f}s")
        print(f"  4 threads: {times[4]:.2f}s")
        if times[1] > 0:
            print(f"  Speedup:   {times[1]/times[4]:.2f}x")


@pytest.mark.requires_julia
class TestThreadCountModes:
    """Test different thread count specification modes."""

    def test_explicit_thread_count(self, test_model_path, julia_available, julia_project_path):
        """Test explicit integer thread count."""
        if not julia_available:
            pytest.skip("Julia not available")

        from ase_ace import ACECalculator
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)

        with ACECalculator(
            test_model_path,
            num_threads=2,
            timeout=120.0,
            julia_project=julia_project_path,
        ) as calc:
            atoms.calc = calc
            E = atoms.get_potential_energy()
            assert np.isfinite(E)

    def test_auto_thread_count(self, test_model_path, julia_available, julia_project_path):
        """Test 'auto' thread count mode."""
        if not julia_available:
            pytest.skip("Julia not available")

        from ase_ace import ACECalculator
        from ase.build import bulk
        import os

        atoms = bulk('Si', 'diamond', a=5.43)

        with ACECalculator(
            test_model_path,
            num_threads='auto',
            timeout=120.0,
            julia_project=julia_project_path,
        ) as calc:
            atoms.calc = calc
            E = atoms.get_potential_energy()
            assert np.isfinite(E)
