"""
Pytest fixtures for ase-ace tests.

This module provides fixtures for testing the ACECalculator.
It requires a pre-fitted test model and Julia installation.
"""

import os
import shutil
import pytest
from pathlib import Path

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "test_model.json"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_julia: marks tests that require Julia"
    )
    config.addinivalue_line(
        "markers", "requires_library: marks tests that require compiled library"
    )


@pytest.fixture(scope="session")
def julia_available():
    """Check if Julia is available."""
    return shutil.which("julia") is not None


@pytest.fixture(scope="session")
def test_model_path():
    """
    Path to pre-fitted test model.

    The test model should be created by running:
        python -m ase_ace.setup_tests

    Or by running the create_test_model() function.
    """
    if not TEST_MODEL_PATH.exists():
        pytest.skip(
            f"Test model not found at {TEST_MODEL_PATH}. "
            "Run 'python -m ase_ace.setup_tests' to create it, "
            "or see tests/README.md for instructions."
        )
    return str(TEST_MODEL_PATH)


@pytest.fixture(scope="session")
def julia_project_path():
    """Path to the Julia project for ase-ace."""
    project_path = Path(__file__).parent.parent / "julia"
    if not (project_path / "Project.toml").exists():
        pytest.skip(f"Julia project not found at {project_path}")
    return str(project_path)


@pytest.fixture
def ace_calculator(test_model_path, julia_available, julia_project_path):
    """
    Create an ACECalculator instance for testing.

    Uses single-threaded mode for deterministic results.
    """
    if not julia_available:
        pytest.skip("Julia not available")

    from ase_ace import ACECalculator

    calc = ACECalculator(
        test_model_path,
        num_threads=1,
        timeout=120.0,  # 2 minutes for Julia startup
        julia_project=julia_project_path,
    )
    yield calc
    calc.close()


@pytest.fixture
def si_diamond():
    """Si diamond unit cell (2 atoms)."""
    from ase.build import bulk
    return bulk('Si', 'diamond', a=5.43)


@pytest.fixture
def si_supercell():
    """Si 2x2x2 supercell (16 atoms)."""
    from ase.build import bulk
    return bulk('Si', 'diamond', a=5.43) * (2, 2, 2)


@pytest.fixture
def perturbed_si():
    """Si supercell with random perturbations."""
    import numpy as np
    from ase.build import bulk

    np.random.seed(42)  # Reproducible
    atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
    return atoms


def create_test_model(output_path: str = None):
    """
    Create a test ACE model using Julia.

    This function calls Julia to fit a small Si model and save it
    as JSON for use in tests.

    Parameters
    ----------
    output_path : str, optional
        Output path for model JSON. Defaults to fixtures/test_model.json.

    Returns
    -------
    str
        Path to created model file.
    """
    import subprocess
    import sys

    if output_path is None:
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(TEST_MODEL_PATH)

    # Julia script to fit and save a small model
    julia_script = '''
    using ACEpotentials
    using Random
    Random.seed!(12345)

    println("Fitting small Si ACE model for testing...")

    # Create a small model
    model = ACEpotentials.ace1_model(
        elements = [:Si],
        order = 2,
        totaldegree = 6,
        rcut = 5.5,
    )

    # Load example dataset
    dataset = ACEpotentials.example_dataset("Si_tiny")
    data = dataset.train

    # Fit the model
    ACEpotentials.acefit!(data, model;
        energy_key = "dft_energy",
        force_key = "dft_force",
        virial_key = "dft_virial",
        weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0)),
        solver = ACEfit.BLR(),
    )

    # Save the model
    ACEpotentials.save_model(model, ARGS[1])
    println("Model saved to: ", ARGS[1])
    '''

    # Get the Julia project path
    julia_project = Path(__file__).parent.parent / "julia"

    cmd = [
        "julia",
        f"--project={julia_project}",
        "-e", julia_script,
        output_path,
    ]

    print(f"Running: {' '.join(cmd[:3])} ...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
    )

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Failed to create test model: {result.stderr}")

    print(result.stdout)
    return output_path


if __name__ == "__main__":
    # Allow running this file directly to create the test model
    path = create_test_model()
    print(f"Test model created at: {path}")
