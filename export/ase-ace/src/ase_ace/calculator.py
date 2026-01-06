"""
ASE Calculator for ACE potentials using Julia/IPICalculator.

This module provides ACECalculator, which manages a Julia subprocess
running an ACE potential and communicates via the i-PI socket protocol.
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
from ase.calculators.calculator import all_changes

from .base import ACECalculatorBase
from .server import JuliaACEServer, find_free_port

logger = logging.getLogger(__name__)


class ACECalculator(ACECalculatorBase):
    """
    ASE calculator for ACE potentials via Julia/IPICalculator.

    This calculator spawns a Julia process that loads an ACE potential
    model and connects as an i-PI driver. The calculator manages the
    subprocess lifecycle automatically.

    Parameters
    ----------
    model_path : str
        Path to the ACE model JSON file (created by ACEpotentials.save_model).
    num_threads : int or str, default='auto'
        Number of Julia threads for parallel evaluation. Use 'auto' to
        use all available CPU cores.
    port : int, default=0
        TCP port for socket communication. Use 0 for automatic assignment.
    unixsocket : str, optional
        Unix socket name (faster than TCP for local connections).
        Mutually exclusive with port.
    timeout : float, default=60.0
        Timeout in seconds for Julia startup and connection.
    julia_executable : str, default='julia'
        Path to Julia executable.
    julia_project : str, optional
        Path to Julia project directory. Defaults to the bundled project.
    log_level : str, default='WARNING'
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Examples
    --------
    Basic usage with context manager (recommended):

    >>> from ase.build import bulk
    >>> from ase_ace import ACECalculator
    >>>
    >>> atoms = bulk('Si', 'diamond', a=5.43)
    >>> with ACECalculator('model.json', num_threads=4) as calc:
    ...     atoms.calc = calc
    ...     energy = atoms.get_potential_energy()
    ...     forces = atoms.get_forces()

    Manual lifecycle management:

    >>> calc = ACECalculator('model.json')
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    >>> calc.close()

    Notes
    -----
    The first calculation may take 5-10 seconds due to Julia's JIT
    compilation. Subsequent calculations are much faster.

    The calculator requires Julia 1.11+ with ACEpotentials.jl and
    IPICalculator.jl installed.
    """

    default_parameters = {}

    def __init__(
        self,
        model_path: str,
        num_threads: Union[int, str] = 'auto',
        port: int = 0,
        unixsocket: Optional[str] = None,
        timeout: float = 60.0,
        julia_executable: str = 'julia',
        julia_project: Optional[str] = None,
        log_level: str = 'WARNING',
    ):
        super().__init__()

        # Configure logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.num_threads = num_threads
        self.port = port
        self.unixsocket = unixsocket
        self.timeout = timeout
        self.julia_executable = julia_executable
        self.julia_project = julia_project

        # Internal state
        self._server: Optional[JuliaACEServer] = None
        self._socket_calc = None
        self._started = False
        self._actual_port: Optional[int] = None

    def _start(self):
        """Start the socket calculator (server) and Julia driver (client)."""
        if self._started:
            return

        logger.info(f"Starting ACECalculator with model: {self.model_path}")

        from ase.calculators.socketio import SocketIOCalculator

        # Determine port/socket for communication
        if self.unixsocket:
            socket_port = None
            socket_name = self.unixsocket
        elif self.port == 0:
            socket_port = find_free_port()
            socket_name = None
        else:
            socket_port = self.port
            socket_name = None

        self._actual_port = socket_port

        # Step 1: Create SocketIOCalculator (ASE server)
        # This creates a listening socket that waits for driver connection
        logger.info(f"Creating SocketIOCalculator (port={socket_port}, unixsocket={socket_name})")

        if socket_name:
            self._socket_calc = SocketIOCalculator(
                unixsocket=socket_name,
                timeout=self.timeout,
            )
        else:
            self._socket_calc = SocketIOCalculator(
                port=socket_port,
                timeout=self.timeout,
            )

        # Step 2: Create and start Julia driver (client)
        # The driver will connect to the socket created above
        self._server = JuliaACEServer(
            model_path=str(self.model_path),
            num_threads=self.num_threads,
            port=socket_port if socket_port else 0,
            unixsocket=socket_name,
            julia_executable=self.julia_executable,
            julia_project=self.julia_project,
        )

        # Start Julia driver - it will connect to SocketIOCalculator
        self._server.start(timeout=self.timeout)

        self._started = True
        logger.info("ACECalculator started successfully")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Perform calculation for the given atoms object.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to calculate.
        properties : list of str
            Properties to calculate ('energy', 'forces', 'stress').
        system_changes : list of str
            Changes since last calculation.
        """
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        # Start server on first calculation
        if not self._started:
            self._start()

        # Delegate to socket calculator
        self._socket_calc.calculate(atoms, properties, system_changes)

        # Copy results
        self.results = self._socket_calc.results.copy()

    def close(self):
        """
        Shutdown the Julia driver and close socket connection.

        This is called automatically when using the calculator as a
        context manager.
        """
        if self._socket_calc is not None:
            try:
                self._socket_calc.close()
            except Exception as e:
                logger.warning(f"Error closing socket calculator: {e}")
            self._socket_calc = None

        if self._server is not None:
            self._server.stop()
            self._server = None

        self._started = False
        logger.info("ACECalculator closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass

    @property
    def actual_port(self) -> Optional[int]:
        """The actual TCP port being used (after start)."""
        return self._actual_port

    # Abstract method implementations (not supported for socket-based calculator)

    @property
    def cutoff(self) -> float:
        """Cutoff radius in Angstroms."""
        raise NotImplementedError(
            "cutoff not available for socket-based ACECalculator. "
            "Use ACEJuliaCalculator or ACELibraryCalculator instead."
        )

    @property
    def species(self) -> List[int]:
        """List of supported atomic numbers."""
        raise NotImplementedError(
            "species not available for socket-based ACECalculator. "
            "Use ACEJuliaCalculator or ACELibraryCalculator instead."
        )

    @property
    def n_basis(self) -> int:
        """Number of basis functions per atom."""
        raise NotImplementedError(
            "n_basis not available for socket-based ACECalculator. "
            "Use ACEJuliaCalculator or ACELibraryCalculator instead."
        )

    def get_descriptors(self, atoms) -> np.ndarray:
        """
        Compute ACE descriptors (basis values) for all atoms.

        Not supported for socket-based ACECalculator.
        Use ACEJuliaCalculator or ACELibraryCalculator instead.
        """
        raise NotImplementedError(
            "get_descriptors() not available for socket-based ACECalculator. "
            "Use ACEJuliaCalculator or ACELibraryCalculator instead."
        )

    def __repr__(self) -> str:
        return f"ACECalculator(model={self.model_path.name}, threads={self.num_threads})"
