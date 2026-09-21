"""
Utility functions for ase-ace.
"""

import json
import logging
import os
import shutil
import subprocess
import warnings
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
        Path to Julia executable.  Defaults to the one juliapkg resolved; naming one
        overrides only the executable.
    julia_project : str, optional
        Path to a Julia project directory.  Defaults to the one juliapkg manages; naming one
        bypasses juliapkg.  See ``ase_ace.server.resolve_julia_env``.

    Returns
    -------
    dict
        ``{package_name: bool}`` for every package in ``juliapkg.json``.
    """
    # One rule for these two arguments, shared with JuliaACEServer and
    # setup_julia_environment: server.resolve_julia_env() is the single implementation, so
    # the three cannot drift.  It always returns a concrete project, so the loop below can
    # always pass --project.
    from .server import resolve_julia_env

    julia_executable, julia_project = resolve_julia_env(julia_executable, julia_project)

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
        Which Julia to instantiate ``julia_project`` with.  On the juliapkg path it is NOT
        honoured -- the operation there *is* a juliapkg resolve, and juliapkg picks its own
        Julia from ``PYTHON_JULIAPKG_EXE``, read once at import -- so passing it alone warns
        and then does the real resolve anyway.  It never causes an unspecified environment
        to be instantiated.
    julia_project : str, optional
        Instantiate this project instead of asking juliapkg.  The full bypass, for a user
        who built their own environment.
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
        # Full bypass: instantiate exactly the project named.  Note this branch is keyed on
        # the PROJECT only.  Keying it on either argument -- which an earlier round did, for
        # consistency with the other two modules -- meant that naming only an executable ran
        # `Pkg.instantiate(); Pkg.precompile()` with no --project at all, i.e. against the
        # user's default GLOBAL Julia environment, and then returned True having installed
        # none of ase-ace's declared packages: a write to a shared environment plus a success
        # report for achieving nothing.  Consistency is not worth that.
        setup_code = '''
        using Pkg
        println("Instantiating project...")
        Pkg.instantiate()
        println("Precompiling...")
        Pkg.precompile()
        println("Setup complete!")
        '''
        # `--project` is unconditional here: inside this branch a project was named, and a
        # Pkg.instantiate() without one would hit the user's global environment.
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
    from .server import juliapkg_environment_error

    try:
        import juliapkg
    except ImportError:
        raise ImportError(
            "juliapkg is required to install the Julia side of ase-ace.  It is a base "
            "dependency; reinstall with `pip install ase-ace`, or pass julia_project= to "
            "instantiate a project you built yourself."
        ) from None

    if julia_executable is not None:
        # Say so rather than appearing to honour it.  There is nothing to override here: the
        # work below IS a juliapkg resolve, and juliapkg chooses its own Julia.
        warnings.warn(
            f"setup_julia_environment(julia_executable={julia_executable!r}) was ignored: "
            "on the juliapkg path the resolve itself chooses the Julia, via the "
            "PYTHON_JULIAPKG_EXE environment variable (read once, before Python starts).  "
            "Pass julia_project= as well to instantiate a project of your own with this "
            "executable instead.  Resolving with juliapkg's Julia.",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        logging.basicConfig()
        logging.getLogger("juliapkg").setLevel(logging.INFO)

    # resolve() directly, rather than calling server.julia_env() first: julia_env() resolves
    # too, so the pre-flight version made a cold machine install Julia and the declared packages
    # and *then* do it again under force=True.  The actionable message comes from the same
    # place either way.
    #
    # OSError is re-raised, not reported as "Setup failed": an unwritable prefix is the one
    # failure with a specific fix, and returning False would throw that message away.
    try:
        juliapkg.resolve(force=True, update=update)
    except OSError as e:
        raise juliapkg_environment_error(e) from e
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

    if verbose:
        print(f"Julia environment ready at {juliapkg.project()}")
        print(f"Using Julia: {juliapkg.executable()}")
    return True
