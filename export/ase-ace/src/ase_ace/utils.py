"""
Utility functions for ase-ace.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def find_julia() -> Optional[str]:
    """
    Find Julia executable.

    Returns
    -------
    str or None
        Path to Julia executable, or None if not found.
    """
    return shutil.which('julia')


def check_julia_version(julia_executable: str = 'julia') -> Tuple[int, int, int]:
    """
    Check Julia version.

    Parameters
    ----------
    julia_executable : str
        Path to Julia executable.

    Returns
    -------
    tuple
        (major, minor, patch) version tuple.

    Raises
    ------
    RuntimeError
        If Julia version cannot be determined.
    """
    try:
        result = subprocess.run(
            [julia_executable, '--version'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Output: "julia version 1.11.0"
        version_str = result.stdout.strip().split()[-1]
        parts = version_str.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2].split('-')[0]))
    except Exception as e:
        raise RuntimeError(f"Failed to get Julia version: {e}")


def check_julia_packages(
    julia_executable: str = 'julia',
    julia_project: Optional[str] = None,
) -> dict:
    """
    Check if required Julia packages are installed.

    Parameters
    ----------
    julia_executable : str
        Path to Julia executable.
    julia_project : str, optional
        Path to Julia project directory.

    Returns
    -------
    dict
        Dictionary with package names as keys and (installed, version) tuples.
    """
    required_packages = ['ACEpotentials', 'IPICalculator', 'AtomsBase']

    project_arg = f"--project={julia_project}" if julia_project else ""

    results = {}
    for pkg in required_packages:
        check_code = f'''
        using Pkg
        try
            @eval using {pkg}
            println("INSTALLED")
        catch
            println("NOT_INSTALLED")
        end
        '''

        cmd = [julia_executable]
        if project_arg:
            cmd.append(project_arg)
        cmd.extend(['-e', check_code])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            installed = 'INSTALLED' in result.stdout
            results[pkg] = installed
        except Exception:
            results[pkg] = False

    return results


def setup_julia_environment(
    julia_executable: str = 'julia',
    julia_project: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Set up Julia environment with required packages.

    Parameters
    ----------
    julia_executable : str
        Path to Julia executable.
    julia_project : str, optional
        Path to Julia project directory.
    verbose : bool
        Print progress messages.

    Returns
    -------
    bool
        True if setup successful.
    """
    from .server import get_julia_project_path

    if julia_project is None:
        julia_project = str(get_julia_project_path())

    setup_code = '''
    using Pkg
    println("Adding ACE registry...")
    try
        Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
    catch e
        println("Registry may already exist: ", e)
    end
    println("Instantiating project...")
    Pkg.instantiate()
    println("Precompiling...")
    Pkg.precompile()
    println("Setup complete!")
    '''

    cmd = [julia_executable, f"--project={julia_project}", '-e', setup_code]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=600,  # 10 minutes for compilation
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Setup timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Setup failed: {e}")
        return False
