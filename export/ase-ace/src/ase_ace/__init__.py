"""
ase-ace: ASE calculators for ACE potentials.

This package provides three ASE-compatible calculators for ACE (Atomic Cluster
Expansion) interatomic potentials:

1. ACECalculator - Uses Julia/IPICalculator via sockets
   - Full Julia runtime with JIT compilation
   - Multi-threading via JULIA_NUM_THREADS
   - Requires Julia installation
   - First call has ~5-10s startup time
   - Install: pip install ase-ace[ipi]

2. ACELibraryCalculator - Uses pre-compiled shared library
   - Instant startup (no JIT)
   - Single-threaded (--trim=safe limitation)
   - Requires deployment package from ACEpotentials.jl
   - No Julia installation needed at runtime
   - Install: pip install ase-ace[lib]

3. ACEJuliaCalculator - Uses JuliaCall for direct integration
   - Full Julia runtime via JuliaCall
   - Multi-threading via JULIA_NUM_THREADS
   - First call has ~10-30s startup time (JIT)
   - Requires Julia managed by juliapkg
   - Install: pip install ase-ace[julia]

Example (socket-based):
    >>> from ase.build import bulk
    >>> from ase_ace import ACECalculator
    >>>
    >>> atoms = bulk('Si', 'diamond', a=5.43)
    >>> with ACECalculator('model.json', num_threads=4) as calc:
    ...     atoms.calc = calc
    ...     energy = atoms.get_potential_energy()

Example (library-based):
    >>> from ase.build import bulk
    >>> from ase_ace import ACELibraryCalculator
    >>>
    >>> calc = ACELibraryCalculator("deployment/lib/libace_model.so")
    >>> atoms = bulk('Si', 'diamond', a=5.43)
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()

Example (JuliaCall-based):
    >>> from ase.build import bulk
    >>> from ase_ace import ACEJuliaCalculator
    >>>
    >>> calc = ACEJuliaCalculator('model.json', num_threads=4)
    >>> atoms = bulk('Si', 'diamond', a=5.43)
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
"""

from .base import ACECalculatorBase
from .calculator import ACECalculator
from .library_calculator import ACELibraryCalculator

# Optional JuliaCall calculator (requires juliacall)
try:
    from .julia_calculator import ACEJuliaCalculator
except ImportError:
    ACEJuliaCalculator = None

__version__ = "0.1.0"
__all__ = [
    "ACECalculatorBase",
    "ACECalculator",
    "ACELibraryCalculator",
    "ACEJuliaCalculator",
]
