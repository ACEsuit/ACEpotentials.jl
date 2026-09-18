"""
Utility functions for ase-ace.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def find_julia() -> Optional[str]:
    """
    Find a Julia executable on ``PATH``.

    This is *not* what the calculators run.  They use the Julia that juliapkg resolved
    against this package's ``juliapkg.json`` (``ase_ace.server.julia_env()``), which may be
    a different build entirely -- a juliaup channel, or one juliapkg downloaded.  This
    function is here for the "is there a Julia on this machine at all?" question.

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


def declared_julia_packages() -> list:
    """
    The Julia packages this package declares, read from the shipped ``juliapkg.json``.

    Derived rather than hardcoded on purpose.  A literal list here would be a *second* copy
    of the dependency set, and the whole point of folding the socket backend onto juliapkg
    was that two copies of that set had silently drifted apart: one said ACEfit and no
    ArgParse, the other said ArgParse and no AtomsCalculators, and nothing compared them.
    """
    decl = Path(__file__).parent / "juliapkg.json"
    with open(decl) as f:
        return sorted(json.load(f).get("packages", {}))


def check_julia_packages(
    julia_executable: Optional[str] = None,
    julia_project: Optional[str] = None,
) -> dict:
    """
    Check whether the declared Julia packages can be loaded.

    Parameters
    ----------
    julia_executable : str, optional
        Path to Julia executable.  Defaults to the one juliapkg resolved.
    julia_project : str, optional
        Path to a Julia project directory.  Defaults to the one juliapkg manages.

    Returns
    -------
    dict
        ``{package_name: bool}`` for every package in ``juliapkg.json``.
    """
    from .server import julia_env

    if julia_executable is None or julia_project is None:
        exe, project = julia_env()
        julia_executable = julia_executable or exe
        julia_project = julia_project or project

    results = {}
    for pkg in declared_julia_packages():
        check_code = f'''
        try
            @eval using {pkg}
            println("INSTALLED")
        catch
            println("NOT_INSTALLED")
        end
        '''

        cmd = [julia_executable, f"--project={julia_project}", '-e', check_code]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            results[pkg] = 'INSTALLED' in result.stdout
        except Exception:
            results[pkg] = False

    return results


def setup_julia_environment(
    julia_executable: Optional[str] = None,
    julia_project: Optional[str] = None,
    verbose: bool = False,
    update: bool = False,
) -> bool:
    """
    Install the Julia side of ase-ace.

    Same name and same meaning as before -- "make the Julia dependencies available" -- but
    a different implementation and a different destination.  It no longer instantiates a
    ``Project.toml`` shipped inside ``site-packages`` (which a normal user cannot write to,
    and which was a second declaration of the dependency set beside ``juliapkg.json``).  It
    asks juliapkg to resolve ``juliapkg.json``, which installs a compatible Julia if needed,
    adds the packages, resolves and precompiles -- under a cross-process file lock, into a
    location juliapkg chooses.

    You rarely need to call this: the calculators resolve on first use.  It exists so that
    the install can be done once, deliberately, at image-build time or before a batch job.

    Parameters
    ----------
    julia_executable : str, optional
        Only meaningful together with ``julia_project``: the explicit-bypass path below.
        juliapkg's own Julia is chosen by ``PYTHON_JULIAPKG_EXE``, read once at import, so
        this argument cannot steer juliapkg.
    julia_project : str, optional
        Instantiate this project instead of asking juliapkg.  The explicit bypass, for a
        user who built their own environment.
    verbose : bool
        Let juliapkg's (or Pkg's) output through to the terminal.
    update : bool
        Ask juliapkg to update the packages as well as resolving them.

    Returns
    -------
    bool
        True if setup succeeded.

    Raises
    ------
    RuntimeError
        If juliapkg cannot create its environment -- almost always a read-only Python
        prefix.  This raises rather than returning False because the message is the useful
        part: it names ``PYTHON_JULIAPKG_PROJECT``, which is the fix.
    """
    if julia_project is not None:
        # Explicit bypass: instantiate the project the caller named, as before.
        setup_code = '''
        using Pkg
        println("Instantiating project...")
        Pkg.instantiate()
        println("Precompiling...")
        Pkg.precompile()
        println("Setup complete!")
        '''
        cmd = [julia_executable or 'julia', f"--project={julia_project}", '-e', setup_code]
        if verbose:
            print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=not verbose,
                text=True,
                timeout=1800,
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    # The normal path: one environment, declared in one file, managed by juliapkg.
    from .server import julia_env  # for the ImportError / read-only-prefix messages

    julia_env()  # resolves, and raises an actionable error if it cannot

    import juliapkg

    if verbose:
        logging.basicConfig()
        logging.getLogger("juliapkg").setLevel(logging.INFO)

    try:
        juliapkg.resolve(force=True, update=update)
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

    if verbose:
        print(f"Julia environment ready at {juliapkg.project()}")
        print(f"Using Julia: {juliapkg.executable()}")
    return True
