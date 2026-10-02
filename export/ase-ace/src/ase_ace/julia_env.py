"""
Where Julia and its packages live, for every backend in this package.

This module used to spawn and manage the Julia driver subprocess for the
socket backend (ASE SocketIOCalculator + IPICalculator over i-PI).  That backend
was removed; what remains here is the juliapkg plumbing the juliacall backend and
``utils`` depend on -- :func:`julia_env`, :func:`resolve_julia_env`,
:func:`get_julia_assets_path` and :func:`juliapkg_environment_error`.

It was called ``server.py`` while it managed that subprocess; the name outlived the
thing it described, so it is now ``julia_env``.

Note the module and one of its functions share a name.  ``from ase_ace.julia_env
import julia_env`` is fine, but binding that function at package level would shadow
the module, so ``__init__`` deliberately imports neither.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


def get_julia_assets_path() -> Path:
    """
    Directory of the Julia *source files* shipped inside this package.

    These are scripts -- ``python_interface.jl`` -- and nothing else.
    This directory used to also hold a ``Project.toml``, so one function answered two
    questions at once: "where is the code?" and "what do I pass to ``--project``?".  Those
    have different answers now.  The code ships in the wheel; the *environment* is created
    at run time by juliapkg, outside the package, from ``ase_ace/juliapkg.json`` -- see
    :func:`julia_env`.

    The path is ``Path(__file__).parent``-relative so that it is correct in an editable
    install and in a wheel alike.  It previously read
    ``Path(__file__).parent.parent.parent / "julia"`` -- the source-checkout layout
    ``export/ase-ace/julia/`` -- which from ``site-packages/ase_ace/`` resolved to
    ``<prefix>/lib/pythonX.Y/julia`` and did not exist.  The Julia-backed calculator was
    therefore broken in every non-editable install; only editable installs were ever tested.
    tests/test_packaging.py installs a built wheel and asserts these assets resolve.
    """
    return Path(__file__).parent / "julia"


def get_julia_project_path() -> Path:
    """
    Deprecated alias of :func:`get_julia_assets_path`.

    It no longer names a Julia project: ``src/ase_ace/julia/Project.toml`` was deleted when
    the socket backend was folded onto juliapkg, leaving ``ase_ace/juliapkg.json`` as this
    package's single declaration of what Julia and which Julia packages it needs.  Callers
    that wanted the directory of shipped ``.jl`` files should use
    :func:`get_julia_assets_path`; callers that wanted something to pass to ``--project``
    should use :func:`julia_env`.  Kept for one release because it was importable from
    user code.
    """
    warnings.warn(
        "ase_ace.julia_env.get_julia_project_path() is deprecated and will be removed in a "
        "future release.  It now returns only the directory of the shipped Julia scripts: "
        "use get_julia_assets_path() for that, or julia_env() for the (executable, project) "
        "pair the calculators actually run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_julia_assets_path()


def juliapkg_environment_error(exc: OSError) -> RuntimeError:
    """
    Turn juliapkg's ``OSError`` into an error a user can act on.

    Shared by every entry point into Julia -- :func:`julia_env` for the socket backend and
    ``utils``, and ``ACEJuliaCalculator._init_julia`` for the juliacall backend, which meets
    the same failure at ``import juliacall`` (juliacall resolves juliapkg at import, and the
    failure is an ``OSError``, not an ``ImportError``).  One message, one place, so the two
    backends cannot drift apart on the one thing a user has to be told.

    The message adapts to whether ``PYTHON_JULIAPKG_PROJECT`` is already set, because
    advising someone to set the variable they have just set is worse than saying nothing.
    """
    configured = os.environ.get("PYTHON_JULIAPKG_PROJECT")
    lines = [f"ase-ace could not create its Julia environment: {exc}"]

    if configured:
        lines += [
            f"PYTHON_JULIAPKG_PROJECT is set to {configured!r}, and that path is not "
            "writable.  Point it somewhere you can write, or fix the permissions:",
            "    export PYTHON_JULIAPKG_PROJECT=$HOME/.julia/environments/ase_ace",
        ]
    else:
        # Be accurate about which installs can hit this.  juliapkg puts the environment in
        # the Python prefix ONLY in a virtualenv or conda environment (juliapkg/state.py:
        # `sys.prefix != sys.base_prefix`, else `CONDA_PREFIX`).  A plain system Python uses
        # <depot>/environments/pyjuliapkg instead -- so "a system-wide install" is NOT a
        # cause of a prefix permission error, and listing it sends people down the wrong
        # path.  What a system Python can hit is an unwritable depot (a read-only $HOME, or
        # JULIA_DEPOT_PATH on a read-only share), which the same variable also solves.
        lines += [
            "In a virtualenv or conda environment juliapkg puts that environment inside the "
            "prefix (<sys.prefix>/julia_env); for a system Python it uses "
            "<JULIA_DEPOT_PATH or ~/.julia>/environments/pyjuliapkg.  Either way the path "
            "above is not writable -- a read-only container image, a shared install serving "
            "several users, or a read-only home directory.",
            "Point juliapkg at a writable directory, e.g.",
            "    export PYTHON_JULIAPKG_PROJECT=$HOME/.julia/environments/ase_ace",
        ]

    lines += [
        "and run once:",
        "    python -c 'from ase_ace.utils import setup_julia_environment; "
        "setup_julia_environment(verbose=True)'",
        "Site administrators building a read-only image: set that variable and run the same "
        "command at build time, then set PYTHON_JULIAPKG_OFFLINE=yes at run time.",
        "Note that PYTHON_JULIAPKG_PROJECT is process-global and shared with juliacall and "
        "every other juliapkg consumer, which is why ase-ace will not set it for you.",
    ]
    return RuntimeError("\n".join(lines))


def julia_env() -> Tuple[str, str]:
    """
    ``(executable, project)`` for the Julia side, as resolved by juliapkg.

    This is the one place ``ase-ace`` decides which Julia runs and which environment it runs
    in, and it delegates both decisions to :mod:`juliapkg`, which reads
    ``ase_ace/juliapkg.json`` (found automatically because ``site-packages`` is on
    ``sys.path`` and ``deps_files()`` descends one level into its entries -- which is why
    that file must stay directly inside the package and not move into ``julia/``).

    Calling this triggers ``juliapkg.resolve()``, which on first use installs a compatible
    Julia and the declared packages.  That takes minutes, so call it lazily -- at
    ``start()``, not at construction.  Subsequent calls are a content-hash check under a
    cross-process file lock and return in milliseconds.

    The juliacall backend does not call this -- juliacall resolves juliapkg itself, at
    import -- but it routes the same failure through :func:`juliapkg_environment_error`.

    Raises
    ------
    ImportError
        If juliapkg is not installed.
    RuntimeError
        If juliapkg cannot create its environment -- almost always an unwritable Python
        prefix.  The message names ``PYTHON_JULIAPKG_PROJECT``, which is the fix; see the
        README's "Where the Julia environment lives".
    """
    try:
        import juliapkg
    except ImportError:
        raise ImportError(
            "juliapkg is required to locate the Julia environment for ase-ace's "
            "Julia-backed calculator.  It is a base dependency of ase-ace; reinstall "
            "with `pip install ase-ace`, or pass julia_executable= and julia_project= "
            "explicitly to bypass juliapkg entirely."
        ) from None

    try:
        return juliapkg.executable(), juliapkg.project()
    except OSError as e:
        raise juliapkg_environment_error(e) from e


def resolve_julia_env(
    julia_executable: Optional[str] = None,
    julia_project: Optional[Union[str, Path]] = None,
) -> Tuple[str, str]:
    """
    ase-ace's ONE rule for the ``julia_executable`` / ``julia_project`` pair.

    Returns a complete ``(executable, project)``; the project is never ``None``, so no
    caller ever runs Julia in an environment nobody chose.

    ==========================  ====================================================
    arguments                   meaning
    ==========================  ====================================================
    neither                     juliapkg decides both.
    ``julia_executable`` only   **Executable override**: run *this* Julia, still
                                against juliapkg's resolved project.
    ``julia_project`` (either)  **Full bypass**: juliapkg is not consulted at all.
                                The executable is the one named, else ``julia``.
    ==========================  ====================================================

    Why the split is asymmetric rather than "either argument bypasses".  That rule was
    tried and was worse: naming only an executable then meant *no project at all*, so
    the socket backend (since removed) ran its driver in Julia's default global
    environment -- where ACEpotentials is not installed -- and
    ``setup_julia_environment(julia_executable=...)`` ran ``Pkg.instantiate()`` against that
    same global environment and returned ``True`` having installed none of ase-ace's
    declared packages.  Consistency across the modules that take these arguments is worth
    having; consistency
    that writes to a shared environment and reports success for achieving nothing is not.

    A project, by contrast, IS a complete answer on its own -- "use this environment I
    built" -- so it stays a full bypass, which is what an HPC user with a hand-built module
    environment wants and it costs them no resolve.

    Note what an executable override cannot do: it cannot change which Julia *juliapkg*
    uses, because juliapkg reads ``PYTHON_JULIAPKG_EXE`` once in ``reset_state()`` at import
    and exposes no setter.  It changes which Julia *ase-ace* then runs against juliapkg's
    project.  For an operation that IS a juliapkg resolve -- ``setup_julia_environment()``
    on its normal path -- there is nothing to override, and that function says so out loud
    rather than appearing to honour the argument.

    One consequence worth knowing: an override runs a Julia that may differ from the one
    juliapkg resolved the project's ``Manifest.toml`` for.  Julia will reuse or re-precompile
    as it sees fit, and will say so if it cannot; that is a legible failure, and a far better
    one than silently using an environment without ACEpotentials in it.
    """
    # Normalise "" (and any other falsy value) to None FIRST, so that every entry point
    # agrees on what "unset" means.  JuliaACEServer.__init__ stores
    # `Path(julia_project) if julia_project else None` -- a truthiness test -- so without
    # this line `julia_project=""` was "unset" through this module but a full bypass emitting
    # a bare `--project=` through a direct call here or through check_julia_packages.
    # Unrealistic input, but two entry points disagreeing about one rule is the thing this
    # function exists to prevent.
    julia_executable = julia_executable or None
    julia_project = julia_project or None

    if julia_project is not None:
        return (julia_executable or "julia", str(julia_project))

    exe, project = julia_env()
    return (julia_executable or exe, project)
