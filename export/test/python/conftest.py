"""
Pytest configuration and fixtures for ACE calculator tests.
"""

import os
import pytest
import numpy as np


@pytest.fixture(scope='session')
def lib_path():
    """Get path to compiled ACE library."""
    path = os.environ.get('ACE_LIB_PATH')
    if not path:
        # Try default location
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'build', 'libace_test.so'
        )
    if not os.path.exists(path):
        pytest.skip(f"ACE library not found at {path}")
    return path


@pytest.fixture(scope='session')
def ace_calculator(lib_path):
    """Create ACE calculator instance."""
    from ase_ace import ACELibraryCalculator
    return ACELibraryCalculator(lib_path)


@pytest.fixture
def si_diamond():
    """Create Si diamond unit cell."""
    from ase.build import bulk
    return bulk('Si', 'diamond', a=5.43)


@pytest.fixture
def si_supercell():
    """Create Si 2x2x2 supercell."""
    from ase.build import bulk
    return bulk('Si', 'diamond', a=5.43) * (2, 2, 2)


@pytest.fixture
def perturbed_si(si_supercell):
    """Create perturbed Si supercell for force testing."""
    np.random.seed(42)
    atoms = si_supercell.copy()
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
    return atoms


@pytest.fixture
def reference_energy_8atom():
    """Reference energy for 8-atom Si cell (stored from Julia tests)."""
    # This will be set by the test suite or read from file
    ref_file = os.path.join(os.path.dirname(__file__), 'reference_data.json')
    if os.path.exists(ref_file):
        import json
        with open(ref_file) as f:
            data = json.load(f)
        return data.get('energy_8atom')
    return None
