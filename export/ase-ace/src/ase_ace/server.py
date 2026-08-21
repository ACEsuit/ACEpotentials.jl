"""
Julia ACE server subprocess management.

This module handles spawning and managing the Julia driver process that
connects to ASE's SocketIOCalculator via the i-PI protocol.
"""

import os
import sys
import time
import socket
import signal
import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_julia_project_path() -> Path:
    """Get the path to the Julia project bundled with this package."""
    return Path(__file__).parent.parent.parent / "julia"


class JuliaACEServer:
    """
    Manages a Julia ACE driver subprocess.

    The server spawns a Julia process running ace_driver.jl which connects
    to ASE's SocketIOCalculator as an i-PI driver.

    Parameters
    ----------
    model_path : str
        Path to the ACE model JSON file.
    num_threads : int or str
        Number of Julia threads. Use 'auto' for automatic detection.
    port : int
        TCP port to connect to. Use 0 for automatic assignment.
    unixsocket : str, optional
        Unix socket name (mutually exclusive with port).
    julia_executable : str
        Path to Julia executable.
    julia_project : str, optional
        Path to Julia project directory. Defaults to bundled project.

    Examples
    --------
    >>> server = JuliaACEServer('model.json', num_threads=4, port=31415)
    >>> server.start()
    >>> # ... use with SocketIOCalculator ...
    >>> server.stop()
    """

    def __init__(
        self,
        model_path: str,
        num_threads: Union[int, str] = 'auto',
        port: int = 0,
        unixsocket: Optional[str] = None,
        julia_executable: str = 'julia',
        julia_project: Optional[str] = None,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.num_threads = num_threads
        self.port = port
        self.unixsocket = unixsocket
        self.julia_executable = julia_executable
        self.julia_project = Path(julia_project) if julia_project else get_julia_project_path()

        self._process: Optional[subprocess.Popen] = None
        self._actual_port: Optional[int] = None

    @property
    def driver_script(self) -> Path:
        """Path to the Julia driver script."""
        return self.julia_project / "ace_driver.jl"

    def _get_thread_count(self) -> int:
        """Resolve thread count, handling 'auto'."""
        if self.num_threads == 'auto':
            return os.cpu_count() or 1
        return int(self.num_threads)

    def _build_command(self, port: int) -> list:
        """Build the Julia command line."""
        cmd = [
            self.julia_executable,
            f"--project={self.julia_project}",
            str(self.driver_script),
            "--model", str(self.model_path),
        ]

        if self.unixsocket:
            cmd.extend(["--unixsocket", self.unixsocket])
        else:
            cmd.extend(["--port", str(port)])

        return cmd

    def _build_env(self) -> dict:
        """Build environment variables for Julia process."""
        env = os.environ.copy()
        env["JULIA_NUM_THREADS"] = str(self._get_thread_count())
        return env

    def start(self, timeout: float = 60.0) -> int:
        """
        Start the Julia driver process.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for server to be ready (seconds).

        Returns
        -------
        int
            The actual port number being used.

        Raises
        ------
        RuntimeError
            If the server fails to start or connect within timeout.
        """
        if self._process is not None:
            raise RuntimeError("Server already running")

        # Determine port to use
        if self.unixsocket:
            actual_port = 0
        elif self.port == 0:
            actual_port = find_free_port()
        else:
            actual_port = self.port

        self._actual_port = actual_port

        # Build command and environment
        cmd = self._build_command(actual_port)
        env = self._build_env()

        logger.info(f"Starting Julia driver: {' '.join(cmd)}")
        logger.info(f"JULIA_NUM_THREADS={env['JULIA_NUM_THREADS']}")

        # Start the process
        try:
            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Julia executable not found: {self.julia_executable}\n"
                "Please install Julia: https://julialang.org/downloads/"
            )

        # Give Julia a moment to start
        time.sleep(0.5)

        # Check if process crashed immediately
        if self._process.poll() is not None:
            stdout, stderr = self._process.communicate()
            raise RuntimeError(
                f"Julia driver failed to start:\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        logger.info(f"Julia driver started (PID: {self._process.pid})")
        return actual_port

    def stop(self, timeout: float = 5.0):
        """
        Stop the Julia driver process.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for graceful shutdown.
        """
        if self._process is None:
            return

        logger.info("Stopping Julia driver...")

        # Try graceful termination first
        self._process.terminate()

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Julia driver did not terminate, killing...")
            self._process.kill()
            self._process.wait()

        self._process = None
        self._actual_port = None

    def is_alive(self) -> bool:
        """Check if the Julia process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_output(self) -> tuple:
        """
        Get stdout/stderr from the Julia process.

        Returns
        -------
        tuple
            (stdout, stderr) strings. Empty if process still running.
        """
        if self._process is None:
            return "", ""

        if self._process.poll() is None:
            return "", ""

        return self._process.communicate()

    @property
    def actual_port(self) -> Optional[int]:
        """The actual port being used (after start())."""
        return self._actual_port

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
