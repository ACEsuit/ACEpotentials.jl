"""
The Julia assets resolve from the INSTALLED package, in a wheel as well as in a checkout.

WHY THIS FILE EXISTS.

This file grew a second job when the socket backend was folded onto juliapkg: it is also the
gate that keeps ase-ace's Julia dependencies declared in exactly ONE place.  See
``TestOneDependencyDeclaration`` at the bottom of tier 1.

`ase_ace.server.get_julia_assets_path()` and `ase_ace.julia_calculator._INTERFACE_PATH` used
to read

    Path(__file__).parent.parent.parent / "julia"

From ``export/ase-ace/src/ase_ace/`` that is ``export/ase-ace/julia/`` -- correct, but only
while the repository is on disk.  The wheel ships ``src/ase_ace`` and nothing else, so from
``site-packages/ase_ace/`` the identical expression climbed out to
``<prefix>/lib/pythonX.Y/julia``, which does not exist.  Two of the three calculators
(``ACECalculator``, socket/i-PI, and ``ACEJuliaCalculator``, JuliaCall) could therefore never
work from a ``pip install ase-ace``; only ``ACELibraryCalculator``, which takes an explicit
path to a ``.so``, was unaffected.

The defect survived the package's entire life because **every** CI job installs it with
``pip install -e``, under which the climb happens to land on the right directory.  A test
suite that only ever runs against an editable install cannot see this class of bug, so a test
that itself only ran under an editable install would repeat the mistake.  Hence the two tiers
below.

TIER 1 -- always runs, no build, no network (``TestPackageRelativeAssets``).
    Asserts the assets exist AND are *inside* the package directory.  "Inside the package" is
    the invariant that actually fails for the old code, and it fails in **every** install
    mode, editable included -- ``parent.parent.parent/julia`` is not under ``ase_ace/``.  This
    is the gate that runs in the default ``pytest`` invocation and on every CI job that
    touches this package.

TIER 2 -- opt-in, builds the artifacts and installs one (``TestBuiltWheel``,
``TestBuiltSdist``, ``TestNonEditableInstall``).
    The end-to-end proof: build the wheel AND the sdist, look inside both, install the wheel
    into a throwaway venv and resolve the assets from there.  Both artifacts are checked
    because they are configured separately -- a `[tool.hatch.build.targets.wheel] exclude`
    leaves the sdist shipping what the wheel refuses, which is exactly what happened here.
    It needs a network-capable build environment and takes ~10-60 s, so it is gated on
    ``ACE_TEST_PACKAGING=1`` rather than slowing every run.  CI sets that variable in the
    ``ase-ace (imports and utils)`` job, which needs neither Julia nor a compiled library --
    see ``.github/workflows/export-ci.yml``.

NEGATIVE CASES.  Each checker is exercised against a layout that is *wrong* -- a synthetic
copy of the pre-fix tree for tier 1, and for tier 2 the real member lists of the two artifacts
this branch actually produced while broken (the wheel with no Julia assets, the sdist with two
Manifests) -- so that no gate here is one of those checks that passes because it asserts
nothing.
"""

import inspect
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# The Julia assets that must travel with the installed package.
#
# ``Manifest.toml`` is deliberately absent: see the exclusion comment in pyproject.toml.
# ``julia/Project.toml`` is absent because it no longer exists -- ``julia/`` is a directory of
# scripts now, not a Julia project, and ``juliapkg.json`` is the single declaration of what
# Julia and which Julia packages this package needs.  ``TestOneDependencyDeclaration`` is what
# stops the second declaration coming back.
REQUIRED_ASSETS = (
    "julia/ace_driver.jl",
    "julia/python_interface.jl",
    "juliapkg.json",
)

# The eight Julia packages the merged declaration must carry.  Written out rather than read
# from juliapkg.json, because a test that reads the file it is checking asserts nothing.
EXPECTED_JULIA_PACKAGES = frozenset({
    "ACEpotentials",
    "ArgParse",
    "AtomsBase",
    "AtomsCalculators",
    "IPICalculator",
    "StaticArrays",
    "Unitful",
    "UnitfulAtomic",
})

# Julia stdlib and self-references that a `using` line may name without a juliapkg.json entry.
JULIA_STDLIB_ALLOWLIST = frozenset({
    "Base", "Core", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils",
    "LinearAlgebra", "Logging", "Markdown", "Pkg", "Printf", "Profile", "Random",
    "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "TOML",
    "UUIDs", "Unicode",
})

# Where each artifact keeps the licence text.  hatchling's `license-files` puts it under the
# dist-info for a wheel; the sdist keeps it at its root.
WHEEL_LICENSE = "ase_ace-0.1.0.dist-info/licenses/LICENSE"

RUN_SLOW = os.environ.get("ACE_TEST_PACKAGING") == "1"
slow = pytest.mark.skipif(
    not RUN_SLOW,
    reason="set ACE_TEST_PACKAGING=1 to build a wheel and install it into a temp venv",
)


# ---------------------------------------------------------------------------------------
# Tier 1: the assets live inside the package directory.


def asset_problems(package_dir, assets=REQUIRED_ASSETS):
    """
    Reasons the assets are not shipped inside ``package_dir``; empty means they are.

    Two distinct failures are reported, because they are distinct bugs:

    * *missing* -- the file is not there at all (what a wheel install saw);
    * *outside the package* -- the path resolves, but to somewhere the wheel does not ship
      (what a source checkout saw, which is why the missing case went unnoticed).

    Returns strings rather than raising so that a caller can report every problem at once and
    so that this function can be run against a deliberately broken tree in the negative test.
    """
    package_dir = Path(package_dir).resolve()
    problems = []
    for rel in assets:
        path = (package_dir / rel).resolve()
        try:
            path.relative_to(package_dir)
        except ValueError:
            problems.append(
                f"{rel}: resolves to {path}, which is OUTSIDE the package directory "
                f"{package_dir} and so is not shipped in the wheel"
            )
            continue
        if not path.exists():
            problems.append(f"{rel}: missing -- expected at {path}")
    return problems


class TestPackageRelativeAssets:
    """The always-on gate.  Cheap, and it fails for the original bug in every install mode."""

    def test_package_ships_its_julia_assets(self):
        import ase_ace

        package_dir = Path(ase_ace.__file__).resolve().parent
        assert asset_problems(package_dir) == []

    def test_julia_assets_path_is_inside_the_package(self):
        import ase_ace
        from ase_ace.server import get_julia_assets_path

        package_dir = Path(ase_ace.__file__).resolve().parent
        assets = get_julia_assets_path().resolve()

        # `relative_to` raises if `assets` is not under the package -- which is exactly what
        # the pre-fix `parent.parent.parent / "julia"` did, editable install or not.
        assets.relative_to(package_dir)
        assert (assets / "ace_driver.jl").exists()
        assert (assets / "python_interface.jl").exists()

    def test_interface_path_is_inside_the_package(self):
        pytest.importorskip("numpy")
        pytest.importorskip("ase")

        import ase_ace
        from ase_ace.julia_calculator import _INTERFACE_PATH

        package_dir = Path(ase_ace.__file__).resolve().parent
        interface = _INTERFACE_PATH.resolve()

        interface.relative_to(package_dir)
        assert interface.exists()

    def test_utils_derives_its_package_list_from_the_declaration(self):
        """
        ``utils`` must not keep its own copy of the dependency set.

        It used to: ``check_julia_packages`` hardcoded
        ``['ACEpotentials', 'IPICalculator', 'AtomsBase']``, a *fourth* list beside
        juliapkg.json, julia/Project.toml and the `using` lines in the .jl files.  Now it
        reads juliapkg.json, so it cannot drift.
        """
        from ase_ace.utils import declared_julia_packages

        assert set(declared_julia_packages()) == EXPECTED_JULIA_PACKAGES

    # -- negative case: prove the checker can fail ---------------------------------------

    def test_checker_rejects_the_pre_fix_layout(self, tmp_path):
        """
        A synthetic copy of the layout this branch replaced: assets one level *above* the
        package.  If `asset_problems` reported this clean it would be worthless.
        """
        pkg = tmp_path / "src" / "ase_ace"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        # assets where they used to live -- `export/ase-ace/julia`, i.e. ../../julia
        old_julia = tmp_path / "julia"
        old_julia.mkdir()
        for name in ("Project.toml", "ace_driver.jl", "python_interface.jl"):
            (old_julia / name).write_text("# placeholder\n")
        (tmp_path / "juliapkg.json").write_text("{}\n")

        problems = asset_problems(pkg)
        assert len(problems) == len(REQUIRED_ASSETS)
        assert all("missing" in p for p in problems)

        # and the climb-out itself is caught as such
        climbed = asset_problems(pkg, assets=("../../julia/Project.toml",))
        assert len(climbed) == 1
        assert "OUTSIDE the package directory" in climbed[0]

    def test_checker_passes_a_correct_layout(self, tmp_path):
        """...and is not simply always-fail."""
        pkg = tmp_path / "ase_ace"
        (pkg / "julia").mkdir(parents=True)
        for rel in REQUIRED_ASSETS:
            (pkg / rel).write_text("# placeholder\n")
        assert asset_problems(pkg) == []


# ---------------------------------------------------------------------------------------
# Tier 1b: one dependency declaration, and it covers what the shipped Julia code uses.


USING_RE = re.compile(r"^\s*(?:using|import)\s+([^\n#]+)", re.MULTILINE)

# `jl.seval('using ACEpotentials')` in julia_calculator.py -- Julia `using` lines that live
# inside Python string literals, so USING_RE (anchored to the start of a line) cannot see
# them.  They are as much a dependency as anything in a .jl file: five of them run at
# ACEJuliaCalculator._init_julia.
#
# TWO KNOWN LIMITS, recorded rather than fixed -- both are complete coverage of what this
# package ships today, and both would silently under-report if that changed:
#   1. a plain quote must follow `seval(`, so an f-string (`seval(f'using {pkg}')`) or a
#      triple-quoted seval is invisible to this;
#   2. `python_sources` below is the single hardcoded file julia_calculator.py, so a future
#      module that sevals would not be scanned at all.
SEVAL_RE = re.compile(r"""seval\(\s*['"]\s*((?:using|import)\s[^'"]+)['"]""")

# Julia source embedded in a triple-quoted block -- `jl.seval('''...''')`, or the
# `julia_script = '''...'''` that conftest.py hands to `julia -e`.  Scanned as Julia, so
# `using X` at the start of a line inside the block is seen.
TRIPLE_RE = re.compile(r"(?:'''|\"\"\")(.*?)(?:'''|\"\"\")", re.DOTALL)

# A Julia package name, as opposed to a sentence that starts with the word "using".
JULIA_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def julia_packages_used(text, kind="julia"):
    """
    Top-level Julia package names that a chunk of source ``using``s or ``import``s.

    Handles the three forms that appear in the shipped files:
    ``using A, B``, ``using A: x, y`` (the package is ``A``) and ``using A.Sub: x``
    (likewise ``A``).  Relative forms (``using .Mod``) are dropped -- they name a module
    defined in the same file, not a dependency.

    With ``kind="python"`` it instead reads `jl.seval` of a `using X` line out of Python,
    ONLY that.  The two are separate because `USING_RE` is anchored to the start of a line
    and would otherwise swallow Python's own `import os` / `import numpy as np`.  Without the
    python mode this checker's name overpromised: it read .jl files only, while five of the
    packages the juliacall backend loads are named in Python string literals in
    julia_calculator.py.  Nothing had drifted, but the invariant was narrower than its name.
    """
    if kind == "python":
        # seval strings, AND Julia embedded in triple-quoted blocks.  A Python file can
        # reach Julia either way: julia_calculator.py uses one-line sevals, conftest.py
        # builds a triple-quoted script for `julia -e`, and the CI workflow does both.
        clauses = [m.split(None, 1)[1] for m in SEVAL_RE.findall(text)]
        for block in TRIPLE_RE.findall(text):
            clauses += USING_RE.findall(block)
    elif kind == "julia":
        clauses = USING_RE.findall(text)
    else:
        raise ValueError(kind)

    found = set()
    for clause in clauses:
        # `using A; f()` -- the load is the half before the semicolon.  Without this split
        # the prose table in test_julia_environment.py, which lists `using ACEpotentials;
        # ACEfit.BLR()` and two variants of it, parsed as four package names with spaces and
        # parentheses in them.
        clause = clause.split(";", 1)[0]
        clause = clause.split(":", 1)[0]
        for name in clause.split(","):
            name = name.strip()
            if not name or name.startswith("."):
                continue
            name = name.split(".", 1)[0]
            # A Julia identifier or it is not a package name.  The scan reads docstrings and
            # comments -- it has to, since a `using` line inside a triple-quoted block is a
            # real dependency -- and English sentences that happen to start with the word
            # "using" or "import" would otherwise be reported as undeclared packages.  This
            # drops only strings no `using` statement can produce, so it cannot hide a
            # dependency; the deliberate negatives, which DO look like package names, are
            # handled by the per-line markers instead.
            if not JULIA_NAME_RE.match(name):
                continue
            found.add(name)
    return found


# The five paths the hand-maintained tuple this replaced named at 6fb3e1a6.  NOT the scan
# input -- the scan input is derived, below.  This is a RATCHET: whatever the derivation
# returns has to cover these, so a later narrowing of the rule cannot quietly take coverage
# back below where it already was.  The derivation is what finds new files; this only stops
# it losing old ones.
GATE_A_FLOOR = (
    "export/ase-ace/src/ase_ace/julia/ace_driver.jl",
    "export/ase-ace/src/ase_ace/julia/python_interface.jl",
    "export/ase-ace/src/ase_ace/julia_calculator.py",
    "export/ase-ace/tests/conftest.py",
    ".github/workflows/export-ci.yml",
)

ASE_ACE_REL = "export/ase-ace"
WORKFLOWS_REL = ".github/workflows"

# A workflow is in scope when it drives THIS package's Julia environment: `seval` is the
# juliacall route (the one that went red), and the `ase_ace`/`ase-ace` mention catches a step
# that drives the package some other way.  Workflows that run Julia against the REPOSITORY's
# own Project.toml -- CI.yml, and export-ci.yml's own `julia --project=export` steps -- are a
# different environment with a different declaration, and are none of gate A's business.
WORKFLOW_IN_SCOPE_RE = re.compile(r"seval|ase[_-]ace")

# Directories a filesystem walk must not descend into.  Used only where `git ls-files` is
# unavailable; a venv or a build tree under export/ase-ace/ would otherwise be scanned.
WALK_SKIP = frozenset({
    ".git", ".venv", "venv", "env", "build", "dist", "__pycache__", ".pytest_cache",
    ".tox", ".mypy_cache", ".ruff_cache", "node_modules", ".eggs", "site-packages",
})


def _walk_files(root, subdirs):
    """Every file under `subdirs`, relative to `root`, skipping build and venv trees."""
    found = []
    for sub in subdirs:
        stack = [Path(root) / sub]
        while stack:
            directory = stack.pop()
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_dir():
                    if path.name in WALK_SKIP or path.name.endswith(".egg-info"):
                        continue
                    stack.append(path)
                elif path.is_file():
                    found.append(path.relative_to(root).as_posix())
    return sorted(found)


def repo_files(root, use_git=True):
    """
    Tracked files under the two directories gate A derives its scan from.

    `git ls-files` is the primary source because "tracked" is the property that matters: an
    untracked scratch file is not something CI runs, and a vendored or copied tree under the
    package would otherwise be scanned.  A new file is therefore invisible to gate A until
    it is `git add`-ed -- which is before it can reach CI, and is the moment the gate starts
    reporting it.  Where git is absent, or `root` is not a work tree (which is how the
    synthetic trees in the tests below are read), it falls back to a pruned filesystem walk,
    so the gate degrades to a slightly broader scan rather than to a skip.
    """
    subdirs = (ASE_ACE_REL, WORKFLOWS_REL)
    if use_git:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--", *subdirs],
                capture_output=True, text=True, check=True, timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return _walk_files(root, subdirs)
        return sorted(p for p in out.split("\0") if p)
    return _walk_files(root, subdirs)


def julia_driving_sources(root, use_git=True):
    """
    Every file in the checkout that runs Julia in ase-ace's juliapkg environment.

    DERIVED, not remembered.  This was a hand-written five-tuple, and a hand-written list is
    worth exactly as much as whoever last remembered it: its one completeness check fired
    when a listed file DISAPPEARED, and nothing fired when a Julia-driving file was ADDED --
    which is precisely the failure the list was added to prevent.  CI went red at c7a471ed
    because `ACEfit` was dropped from juliapkg.json while `.github/workflows/export-ci.yml`
    still handed a `using ACEfit` line to `jl.seval`, and the gate was looking only at
    julia_calculator.py.

    Three parts, unioned:

      * every ``.jl`` under ``export/ase-ace/`` -- today that is the two shipped driver
        scripts in ``src/ase_ace/julia/``, which run in the environment juliapkg builds by
        definition, and the rule does not stop covering them if a third lands elsewhere in
        the package;
      * every ``.py`` under ``export/ase-ace/`` -- the package, its tests, its examples and
        its benchmarks.  NOT "every .py that imports juliacall or juliapkg": that narrower
        rule sounds right and misses ``benchmark_interfaces.py``, which reaches Julia through
        ``ase_ace.server.julia_env()`` and imports neither, and whose Julia script carried an
        undeclared ``using JSON`` for as long as nothing was looking.  A file that touches no
        Julia contributes no names, so breadth costs nothing here;
      * every workflow that drives this package -- see ``WORKFLOW_IN_SCOPE_RE``.

    Deliberately OUT of scope: ``benchmark/*.jl``, ``docs/src/tutorials/asp.jl`` and the rest
    of the repository's own Julia.  Those run under the repository's Project.toml, where
    ACEfit *is* a direct dependency; scanning them against ase-ace's declaration would report
    a conflict that does not exist.  The scope is one environment, not one file extension.

    Returns ``((path relative to root, kind), ...)``, sorted.
    """
    pairs = []
    for rel in repo_files(root, use_git=use_git):
        if rel.startswith(ASE_ACE_REL + "/") and rel.endswith(".jl"):
            pairs.append((rel, "julia"))
        elif rel.startswith(ASE_ACE_REL + "/") and rel.endswith(".py"):
            pairs.append((rel, "python"))
        elif rel.startswith(WORKFLOWS_REL + "/") and rel.endswith((".yml", ".yaml")):
            text = (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
            if WORKFLOW_IN_SCOPE_RE.search(text):
                pairs.append((rel, "python"))
    return tuple(sorted(pairs))


def installed_julia_driving_sources(package_dir):
    """
    The same derivation against an installed package, where the checkout is not on disk.

    Only the wheel's own contents exist there: the shipped ``julia/*.jl`` and the package's
    own ``.py`` modules.  The tests, the benchmarks and the workflow are not installed, so an
    installed run checks less than a checkout run does -- as it always has; the floor ratchet
    below is a checkout-only assertion for that reason.
    """
    pairs = [(f"julia/{p.name}", "julia")
             for p in sorted((Path(package_dir) / "julia").glob("*.jl"))]
    pairs += [(p.name, "python") for p in sorted(Path(package_dir).glob("*.py"))]
    return tuple(pairs)


# ---- deliberate negatives opt out one line at a time, with a reason ----------------------
#
# Some `using` lines are fixtures rather than dependencies.  This file's own proof that the
# checker reports an undeclared package feeds it a package that does not exist;
# test_julia_environment.py loads ACEfit on purpose, in a clean environment, to prove it
# CANNOT be loaded there.  A derivation that read those naively would go red on the negative
# tests -- and that, not an oversight, is why the scan input used to be a hand list with
# those files left out of it.  Whole files opting out invisibly is the thing being replaced,
# so a negative now opts out one LINE at a time, and has to say why, on that line:
#
#     probe = "using Nowhere"  # gate-A-negative: fixture for the checker's own failure path
#
# The reason is mandatory.  A marker with nothing after the colon is an exemption with no
# argument behind it, which is how a per-line opt-out decays back into the invisible
# whole-file kind, so `gate_a_marker_problems` fails the gate on one.
GATE_A_MARKER = "gate-A-" "negative:"  # written in halves: this line is not itself a marker
# Ten characters is not a quality bar, it is a floor under "" and "x": enough that whoever
# adds a marker has to type a clause rather than a shrug.
MIN_GATE_A_REASON = 10


def gate_a_marker(line):
    """The reason on this line's negative marker: None if there is none, '' if it is empty."""
    at = line.find(GATE_A_MARKER)
    if at < 0:
        return None
    return line[at + len(GATE_A_MARKER):].strip()


def gate_a_marker_problems(rel, text):
    """Reasons the negative markers in `text` are not usable; empty means they are."""
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        reason = gate_a_marker(line)
        if reason is None:
            continue
        if len(reason) < MIN_GATE_A_REASON:
            problems.append(
                f"{rel}:{number}: a gate-A negative marker with no reason ({reason!r}).  Say "
                f"what the line is proving, or drop the marker and declare the package -- an "
                f"exemption nobody had to justify is how this gate stopped seeing whole "
                f"files in the first place"
            )
        if "'''" in line or '"""' in line:
            problems.append(
                f"{rel}:{number}: a gate-A negative marker on a line that opens or closes a "
                f"triple-quoted block.  Dropping this line moves the block's boundaries and "
                f"silently changes what is scanned; mark the `using` line inside the block "
                f"instead"
            )
    return problems


def drop_gate_a_negatives(text):
    """`text` with every line carrying a negative marker blanked, line numbering intact."""
    return "".join(
        "\n" if gate_a_marker(line.rstrip("\n")) is not None else line
        for line in text.splitlines(keepends=True)
    )


def repo_root():
    """The repository root, or None when running against an installed package."""
    root = PACKAGE_ROOT.parent.parent
    return root if (root / ".github").is_dir() else None


def undeclared_packages(sources, declared, python_sources=()):
    """
    Package names loaded by the sources but absent from `declared`; empty means consistent.

    `sources` is an iterable of Julia source strings and `python_sources` of Python source
    strings (scanned for `jl.seval` of a `using X` line only); `declared` is the names in
    juliapkg.json.  Returns a sorted list so the failure message names them.
    """
    used = set()
    for text in sources:
        used |= julia_packages_used(text, kind="julia")
    for text in python_sources:
        used |= julia_packages_used(text, kind="python")
    return sorted(used - set(declared) - JULIA_STDLIB_ALLOWLIST)


def project_toml_problems(package_dir):
    """
    Reasons `package_dir` declares its Julia dependencies more than once; empty means once.
    """
    stray = Path(package_dir) / "julia" / "Project.toml"
    if stray.exists():
        return [
            f"{stray}: a second declaration of the Julia dependency set.  juliapkg.json is "
            f"the only one; a Project.toml here drifted out of agreement with it once "
            f"already (it said ArgParse/IPICalculator/UnitfulAtomic and no AtomsCalculators, "
            f"while juliapkg.json said the reverse) and nothing compared them."
        ]
    return []


JULIAPKG_FLOOR_RE = re.compile(r'"juliapkg\s*>=\s*([0-9]+(?:\.[0-9]+)*)"')

# The oldest juliapkg this package's code can actually call.  Not a "keep it fresh" number:
# each entry below is an API `ase-ace` uses that is absent from older published sdists.
MIN_JULIAPKG = (0, 1, 22)
# name -> (probe, what depends on it).  The reason travels with the probe so a failure says
# what is actually broken instead of "an API is missing".
JULIAPKG_APIS_WE_CALL = {
    "resolve(update=)": (
        lambda jp: "update" in inspect.signature(jp.deps.resolve).parameters,
        "utils.setup_julia_environment() passes update=; resolve() has no such parameter "
        "up to and including 0.1.12, so the README's headline install command raises "
        "TypeError there",
    ),
    "executable": (
        lambda jp: callable(jp.executable),
        "server.julia_env() calls it to pick the Julia both backends run",
    ),
    "project": (
        lambda jp: callable(jp.project),
        "server.julia_env() calls it to pick the environment both backends run in",
    ),
    "deps.find_requirements": (
        lambda jp: callable(jp.deps.find_requirements),
        "the tier-2 installed-wheel probe calls it to check the merged declaration",
    ),
    "deps._UUID_RE": (
        lambda jp: jp.deps._UUID_RE is not None,
        "test_juliapkg_json_is_well_formed imports it to validate the UUIDs we ship "
        "(private upstream symbol -- if it is merely RENAMED, fix this test rather than "
        "assuming ase-ace is broken)",
    ),
    "deps.FileLock": (
        lambda jp: jp.deps.FileLock is not None,
        # Deliberately kept despite being a private symbol ase-ace never calls itself.  What
        # depends on it is a CLAIM -- that folding onto juliapkg inherited a cross-process
        # lock for concurrent resolves -- and that claim was false at the old `>=0.1.10`
        # floor, where juliapkg had no FileLock at all and depended only on
        # semantic_version.  A gate on a claim the README and two commit messages make is
        # worth a small false-alarm risk; the wording says which kind of failure this is.
        "ase-ace does not call it, but relies on juliapkg holding it during resolve() -- "
        "this package's concurrency claim.  Absent entirely in 0.1.12 and earlier.  Private "
        "upstream symbol: if it is merely RENAMED, fix this probe, do not assume breakage",
    ),
}


def juliapkg_floor_problems(pyproject_text, installed=None):
    """
    Reasons the declared juliapkg floor does not cover the API this package calls.

    `installed` is the imported juliapkg module, or None to skip the runtime half.  Returns
    strings so the negative test can drive it with a deliberately low floor.
    """
    problems = []
    floors = JULIAPKG_FLOOR_RE.findall(pyproject_text)
    if not floors:
        return ["no `juliapkg>=X` requirement found in pyproject.toml"]
    for floor in floors:
        parts = tuple(int(x) for x in floor.split("."))
        parts += (0,) * (3 - len(parts))
        if parts < MIN_JULIAPKG:
            problems.append(
                f"juliapkg>={floor}: below {'.'.join(map(str, MIN_JULIAPKG))}, which admits "
                f"versions this code cannot call.  Measured against the published sdists: "
                f"0.1.12 has `resolve(force=False, dry_run=False)` -- no `update=`, so "
                f"setup_julia_environment() raises TypeError -- and no `_UUID_RE` and no "
                f"`FileLock` at all, so the cross-process lock this package relies on for "
                f"concurrent resolves does not exist; `find_requirements` is missing "
                f"earlier still (0.1.10, 0.1.11), and 0.1.10's sole dependency is "
                f"semantic_version"
            )
    if installed is not None:
        for name, (probe, why) in sorted(JULIAPKG_APIS_WE_CALL.items()):
            try:
                ok = probe(installed)
            except Exception as e:  # AttributeError on an old version
                ok = False
                name = f"{name} ({type(e).__name__})"
            if not ok:
                problems.append(
                    f"installed juliapkg does not provide {name}: {why}"
                )
    return problems


class TestOneDependencyDeclaration:
    """
    ase-ace declares its Julia dependencies in exactly one file, and that file is complete.

    Before this branch there were two: ``src/ase_ace/juliapkg.json`` for the juliacall
    backend and ``src/ase_ace/julia/Project.toml`` for the socket backend, which spawned
    ``julia --project=<package dir>`` and instantiated inside site-packages.  They disagreed
    in both directions and about the Julia version.  These tests are cheap, offline, and
    would have caught that.
    """

    def test_the_julia_project_toml_is_gone(self):
        import ase_ace

        package_dir = Path(ase_ace.__file__).resolve().parent
        assert project_toml_problems(package_dir) == []

    def test_declaration_checker_rejects_a_reinstated_project_toml(self, tmp_path):
        """Negative case: the checker must fail if the second mechanism comes back."""
        pkg = tmp_path / "ase_ace"
        (pkg / "julia").mkdir(parents=True)
        assert project_toml_problems(pkg) == []

        (pkg / "julia" / "Project.toml").write_text("[deps]\n")
        problems = project_toml_problems(pkg)
        assert len(problems) == 1
        assert "second declaration" in problems[0]

    def test_every_shipped_julia_using_is_declared(self):
        """
        Every package anything in this package loads is in juliapkg.json.

        This is the invariant the two declarations broke: ``ace_driver.jl`` does
        ``using ArgParse``, ``using IPICalculator`` and ``using UnitfulAtomic``, and for the
        package's whole life none of the three was in juliapkg.json -- so the socket backend
        was unusable in any environment juliapkg had built.

        It is also the invariant that went red in CI after ``ACEfit`` was dropped from the
        declaration: the workflow's fixture-fitting step sevaled a ``using ACEfit`` line and
        this gate was only looking at ``julia_calculator.py``.

        What is scanned is now DERIVED -- see ``julia_driving_sources`` -- so a new ``.jl``
        script, a new module, or a new workflow step that sevals is covered the moment it
        lands, rather than when somebody remembers to extend a tuple.  The failure names the
        file each undeclared package came from, because with a derived scan that is no longer
        obvious.
        """
        import ase_ace

        package_dir = Path(ase_ace.__file__).resolve().parent
        declared = json.loads((package_dir / "juliapkg.json").read_text())["packages"]

        root = repo_root()
        base = root if root is not None else package_dir
        scanned = (julia_driving_sources(root) if root is not None
                   else installed_julia_driving_sources(package_dir))
        assert scanned, f"the derivation found nothing to scan under {base}"

        marker_problems, problems = [], []
        for rel, kind in scanned:
            text = (base / rel).read_text(encoding="utf-8", errors="replace")
            marker_problems += gate_a_marker_problems(rel, text)
            used = julia_packages_used(drop_gate_a_negatives(text), kind=kind)
            problems += [f"{rel}: {name}" for name
                         in sorted(used - set(declared) - JULIA_STDLIB_ALLOWLIST)]

        assert marker_problems == [], "\n".join(marker_problems)
        assert problems == [], (
            f"loaded but not declared in juliapkg.json:\n  " + "\n  ".join(problems) +
            f"\nscanned {len(scanned)} files: {[rel for rel, _ in scanned]}"
        )

    def test_the_derived_scan_still_covers_what_the_hand_list_named(self):
        """
        The ratchet: the derivation may grow, and may not shrink below the old hand list.

        ``GATE_A_FLOOR`` is not the scan input -- if it were, this would be the hand list
        again under a new name.  It is the low-water mark: a future tightening of
        ``julia_driving_sources`` that dropped, say, the workflow or conftest.py would take
        the gate back to exactly the coverage that let CI go red, and this fails when it
        does.
        """
        root = repo_root()
        if root is None:
            # An installed package has no tests, no benchmarks and no workflow.  The floor
            # that still applies is the part of it the wheel actually ships.
            import ase_ace

            package_dir = Path(ase_ace.__file__).resolve().parent
            found = {rel for rel, _ in installed_julia_driving_sources(package_dir)}
            assert {"julia/ace_driver.jl", "julia/python_interface.jl",
                    "julia_calculator.py"} <= found, found
            return

        found = {rel for rel, _ in julia_driving_sources(root)}
        missing = [rel for rel in GATE_A_FLOOR if rel not in found]
        assert missing == [], (
            f"the derivation no longer covers {missing}, which the hand-maintained list it "
            f"replaced did cover.  Widen the rule in julia_driving_sources -- do not narrow "
            f"the floor"
        )

        # ...and the other half of the ratchet: it must not GROW into the repository's own
        # Julia.  `benchmark/*.jl` and `docs/src/tutorials/asp.jl` do `using ACEfit`, legally
        # -- they run under the repository's Project.toml, which has ACEfit as a direct
        # dependency.  Scanning them against ase-ace's declaration would report a conflict
        # that does not exist, and the honest fix would then look like declaring ACEfit.
        strays = [rel for rel in found
                  if not rel.startswith((ASE_ACE_REL + "/", WORKFLOWS_REL + "/"))]
        assert strays == [], strays

    def test_the_derivation_sees_a_file_the_hand_list_could_not(self, tmp_path):
        """
        The hole this replaced, as a test: a new Julia-driving file is scanned unasked.

        Driven against a synthetic tree rather than the real one, because the point is what
        happens to a file that does not exist yet.  Three arrivals, each of which the
        five-entry tuple would have missed in silence: a new ``.jl`` beside the shipped
        drivers, a new module in the package, and a new workflow step that sevals.  Run
        twice -- once as a git work tree, once as plain directories -- so the fallback in
        ``repo_files`` is exercised as well as the `git ls-files` path.
        """
        root = tmp_path
        julia = root / ASE_ACE_REL / "src" / "ase_ace" / "julia"
        julia.mkdir(parents=True)
        (julia / "ace_driver.jl").write_text("using ACEpotentials\n")
        (julia / "scratch.jl").write_text("using Nonexistent\n")
        (julia.parent / "julia_calculator.py").write_text("x = 1\n")
        (julia.parent / "later_module.py").write_text(
            "def f(jl):\n    jl.seval('using Missing2')\n"  # gate-A-negative: a fixture tree
        )
        workflows = root / WORKFLOWS_REL
        workflows.mkdir(parents=True)
        (workflows / "export-ci.yml").write_text(
            "run: python -c \"jl.seval('using Missing3')\"\n"  # gate-A-negative: a fixture tree
        )
        (workflows / "unrelated.yml").write_text("run: julia --project=. -e 'using Whatever'\n")

        for use_git in (False, True):
            if use_git:
                try:
                    subprocess.run(["git", "init", "-q", str(root)], check=True,
                                   capture_output=True, text=True)
                    subprocess.run(["git", "-C", str(root), "add", "--",
                                    ASE_ACE_REL, WORKFLOWS_REL], check=True,
                                   capture_output=True, text=True)
                except (OSError, subprocess.SubprocessError):
                    # No usable git here, so `repo_files` would fall back to the walk that
                    # the use_git=False pass has already driven.  Nothing left to prove.
                    continue
            found = julia_driving_sources(root, use_git=use_git)
            rels = [rel for rel, _ in found]
            assert f"{ASE_ACE_REL}/src/ase_ace/julia/scratch.jl" in rels, rels
            assert f"{ASE_ACE_REL}/src/ase_ace/later_module.py" in rels, rels
            assert f"{WORKFLOWS_REL}/export-ci.yml" in rels, rels
            # ...and a workflow that runs Julia against some OTHER project is not dragged in.
            assert f"{WORKFLOWS_REL}/unrelated.yml" not in rels, rels

            names = set()
            for rel, kind in found:
                names |= julia_packages_used((root / rel).read_text(), kind=kind)
            assert {"Nonexistent", "Missing2", "Missing3"} <= names, names
            assert "Whatever" not in names, names

    def test_a_negative_marker_must_carry_a_reason(self):
        """
        An exemption nobody had to justify is the failure mode being designed out.

        The marker is written in halves here for the same reason it is in the constant: a
        literal one with an empty reason, sitting in a file the gate scans, would fail the
        gate on its own negative test.
        """
        marker = "# " + GATE_A_MARKER
        assert gate_a_marker_problems("f.py", f"probe = 'using Nowhere'  {marker}\n") != []
        assert gate_a_marker_problems("f.py", f"probe = 'using Nowhere'  {marker}  \n") != []
        assert gate_a_marker_problems("f.py", f"probe = 'using Nowhere'  {marker} eh\n") != []

        good = f"probe = 'using Nowhere'  {marker} fixture for the failure path\n"
        assert gate_a_marker_problems("f.py", good) == []
        assert julia_packages_used(drop_gate_a_negatives(good), kind="julia") == set()

        # A marker on a line that delimits a triple-quoted block would move the block's
        # boundaries when the line is dropped, changing what is scanned somewhere else.
        fence = "probe = '" + "''" + f"  {marker} a reason long enough\n"
        assert [p for p in gate_a_marker_problems("f.py", fence) if "triple" in p]

    def test_a_negative_marker_exempts_only_its_own_line(self):
        """A per-line opt-out that took its neighbours with it would be the old hole again."""
        text = (
            f"using Real\n"
            f"using Fixture  # {GATE_A_MARKER} a fixture, and this reason is long enough\n"
            f"using AlsoReal\n"
        )
        assert julia_packages_used(drop_gate_a_negatives(text)) == {"Real", "AlsoReal"}
        assert drop_gate_a_negatives(text).count("\n") == 3

    def test_using_checker_rejects_the_pre_fold_declaration(self):
        """
        Negative case, twice over.

        First a synthetic source naming a package nobody declared; then the declaration this
        package actually shipped before the fold, checked against the `using` lines it
        actually shipped alongside it.  The second is the one that matters: it reports the
        exact three names the fold added, from the real files, so this gate is pinned to a
        failure that really happened rather than to an invented one.
        """
        declared = dict.fromkeys(EXPECTED_JULIA_PACKAGES)
        assert undeclared_packages(["using Nonexistent\n"], declared) == ["Nonexistent"]

        # The pre-fold src/ase_ace/juliapkg.json, verbatim.
        pre_fold = ["ACEpotentials", "ACEfit", "AtomsBase", "AtomsCalculators",
                    "Unitful", "StaticArrays"]
        # The pre-fold ace_driver.jl / python_interface.jl `using` blocks, verbatim.
        pre_fold_sources = [
            "using ArgParse\nusing ACEpotentials\nusing IPICalculator\n"
            "using AtomsBase\nusing Unitful\nusing UnitfulAtomic\n",
            "using ACEpotentials\nusing ACEpotentials: site_descriptors\n"
            "using ACEpotentials.Models: energy_forces_virial_basis, cutoff_radius, "
            "length_basis\nusing AtomsBase\nusing AtomsCalculators\nusing Unitful\n"
            "using Unitful: ustrip\nusing StaticArrays\n",
        ]
        assert undeclared_packages(pre_fold_sources, pre_fold) == [
            "ArgParse", "IPICalculator", "UnitfulAtomic",
        ]

        # ...and the parser must not invent dependencies out of the qualified and relative
        # forms that appear in those same files.
        assert julia_packages_used("using ACEpotentials.Models: cutoff_radius\n") == {
            "ACEpotentials"
        }
        assert julia_packages_used("using .ACEPythonInterface\n") == set()

        # ...and it must see the Python-embedded form, or the four lines added above are
        # decoration.
        one_seval = "jl.seval('using StaticArrays')\n"  # gate-A-negative: a fixture line
        assert julia_packages_used(one_seval, kind="python") == {"StaticArrays"}
        nowhere = "jl.seval('using Nowhere')\n"  # gate-A-negative: the checker's own probe
        assert undeclared_packages([], declared, python_sources=[nowhere]) == ["Nowhere"]
        # ...without mistaking Python's own imports for Julia dependencies, which is why the
        # two modes are separate rather than one regex over everything.
        assert julia_packages_used(
            "import os\nimport numpy as np\nfrom pathlib import Path\n", kind="python"
        ) == set()

        # ...and it must not invent dependencies out of English.  The scan reads docstrings
        # and comments on purpose, so prose beginning with the word "using" reaches the
        # parser; every one of these came out of a real file in this repository before the
        # name was required to be an identifier.
        prose = (
            "using ACEfit                        -> ArgumentError: not found\n"
            "import -- but it routes the same failure through juliapkg_environment_error\n"
        )
        assert julia_packages_used(prose) == set()
        # `using A; f()` is a load of A and a call, not a package called "A; f()".
        assert julia_packages_used("using ACEpotentials; ACEfit.BLR()\n") == {
            "ACEpotentials"
        }

    def test_juliapkg_json_is_well_formed(self):
        """
        juliapkg must be able to parse what we ship: real UUIDs and a real Julia compat.

        juliapkg is a base dependency now, so this is imported rather than importorskip'd --
        if it is missing, that is itself the failure.
        """
        import ase_ace
        from juliapkg.compat import Compat
        from juliapkg.deps import _UUID_RE

        package_dir = Path(ase_ace.__file__).resolve().parent
        decl = json.loads((package_dir / "juliapkg.json").read_text())

        assert set(decl["packages"]) == EXPECTED_JULIA_PACKAGES
        for name, spec in decl["packages"].items():
            assert _UUID_RE.match(spec["uuid"]), f"{name}: {spec['uuid']} is not a UUID"

        compat = Compat.parse(decl["julia"])
        assert str(compat)
        # `~1.11, ~1.12` and not `1.11, 1.12`: a comma is a union of CARET ranges, so the
        # bare form means [1.11, 2.0) and admits 1.13 -- which ACEpotentials does not
        # support (see commit 466b58f4).  juliapkg resolves with upgrade=True, i.e. the
        # newest compatible Julia juliaup offers, so this distinction decides what a user
        # actually runs.
        from juliapkg.compat import Version

        assert Version.parse("1.11.7") in compat
        assert Version.parse("1.12.6") in compat
        assert Version.parse("1.13.0") not in compat
        assert Version.parse("1.10.10") not in compat

    def test_the_bypass_rule_has_one_implementation(self, monkeypatch):
        """
        The `julia_executable` / `julia_project` rule is implemented ONCE and obeyed thrice.

        It spans `JuliaACEServer`, `utils.check_julia_packages` and
        `utils.setup_julia_environment`, and an earlier round had all three disagreeing:
        one treated either argument as a total bypass, one resolved for whichever argument
        was absent, one silently ignored the executable.  Worse, "either argument bypasses"
        made `setup_julia_environment(julia_executable=...)` run `Pkg.instantiate()` with no
        `--project` -- against the user's *global* Julia environment -- and return True
        having installed none of ase-ace's packages.  This is the gate on that not coming
        back, and it needs no Julia: `julia_env` is stubbed.
        """
        from ase_ace import server, utils

        calls = []

        def fake_julia_env():
            calls.append("resolved")
            return ("/juliapkg/julia", "/juliapkg/project")

        monkeypatch.setattr(server, "julia_env", fake_julia_env)

        # 1. neither: juliapkg decides both.
        assert server.resolve_julia_env() == ("/juliapkg/julia", "/juliapkg/project")
        # 2. executable only: OVERRIDE -- their Julia, juliapkg's project.  Not "no project".
        assert server.resolve_julia_env("/my/julia", None) == (
            "/my/julia",
            "/juliapkg/project",
        )
        # 3. project only: FULL BYPASS -- juliapkg is not consulted at all.
        before = len(calls)
        assert server.resolve_julia_env(None, "/my/proj") == ("julia", "/my/proj")
        assert len(calls) == before, "a named project must cost no juliapkg resolve"
        # 4. both: full bypass, both honoured.
        assert server.resolve_julia_env("/my/julia", "/my/proj") == (
            "/my/julia",
            "/my/proj",
        )

        # The project is NEVER None or empty, for any combination -- that is the invariant
        # that stops anything running Julia in an environment nobody chose.  "" is in the
        # list because JuliaACEServer.__init__ stores the project with a truthiness test
        # while this function keys on `is not None`; without normalisation the two entry
        # points disagreed, and `julia_project=""` meant "unset" through the server but a
        # bypass emitting a bare `--project=` through a direct call.
        for exe, proj in [(None, None), ("/my/julia", None), (None, "/my/proj"),
                          ("/my/julia", "/my/proj"), ("", ""), (None, ""), ("", None)]:
            assert server.resolve_julia_env(exe, proj)[1] not in (None, "")
        assert server.resolve_julia_env("", "") == ("/juliapkg/julia", "/juliapkg/project")

        # ...and the two consumers route through that one function rather than reimplementing
        # it.  `JuliaACEServer` resolves in start(), not __init__.
        assert "resolve_julia_env" in inspect.getsource(server.JuliaACEServer.start)
        assert "resolve_julia_env" in inspect.getsource(utils.check_julia_packages)


    def test_setup_julia_environment_never_instantiates_an_unnamed_environment(
        self, monkeypatch
    ):
        """
        `setup_julia_environment` is the deliberate exception to the rule above, and this is
        a BEHAVIOURAL gate on it -- no source grepping, which a previous draft of this test
        did and which quietly passed the regression because the same `if` appears twice in
        the function.

        Its juliapkg path IS a resolve, so there is no executable to override.  Keying its
        bypass on either argument -- as an earlier round did -- meant
        `setup_julia_environment(julia_executable=...)` ran `Pkg.instantiate()` with no
        `--project`, against the user's global Julia environment, and returned True having
        installed nothing this package declares.  Two invariants: an executable alone must
        still RESOLVE (and say the argument was ignored), and no subprocess may ever run
        without an explicit `--project`.
        """
        from ase_ace import utils

        resolves, runs = [], []

        class FakeJuliapkg:
            def resolve(self, **kw):
                resolves.append(kw)

            def project(self):
                return "/juliapkg/project"

            def executable(self):
                return "/juliapkg/julia"

        class FakeCompleted:
            returncode = 0

        monkeypatch.setitem(sys.modules, "juliapkg", FakeJuliapkg())
        monkeypatch.setattr(
            utils.subprocess, "run", lambda cmd, **kw: runs.append(cmd) or FakeCompleted()
        )

        # 1. executable alone: resolves, warns, and runs NO subprocess of its own.
        with pytest.warns(RuntimeWarning, match="was ignored"):
            assert utils.setup_julia_environment(julia_executable="/my/julia") is True
        assert len(resolves) == 1, "an executable-only call must still do the real resolve"
        assert runs == [], (
            "an executable-only call must not instantiate anything itself -- that is how it "
            "used to write to the user's global environment"
        )

        # 2. project named: full bypass, one subprocess, and it carries --project.
        resolves.clear()
        assert utils.setup_julia_environment(julia_project="/my/proj") is True
        assert resolves == [], "a named project must cost no juliapkg resolve"
        assert len(runs) == 1
        assert "--project=/my/proj" in runs[0]

        # 3. the invariant, over every form that reaches a subprocess: never project-less.
        runs.clear()
        utils.setup_julia_environment(julia_executable="/my/julia", julia_project="/my/proj")
        assert runs[0][0] == "/my/julia"
        assert any(a.startswith("--project=") for a in runs[0]), (
            "no Pkg.instantiate() may run without an explicit --project"
        )

    def test_bypass_rule_checker_rejects_the_either_argument_form(self, monkeypatch):
        """
        Negative case: the "either argument bypasses" rule this replaced must not pass.

        Reimplemented here exactly as it was written, and driven through the same assertions
        the real rule is held to.  It fails on the one that matters -- an executable-only
        call returning no project.
        """
        def either_argument_bypasses(julia_executable=None, julia_project=None):
            if julia_executable is None and julia_project is None:
                return ("/juliapkg/julia", "/juliapkg/project")
            return (julia_executable or "julia", julia_project)  # project may be None!

        assert either_argument_bypasses()[1] is not None
        assert either_argument_bypasses(None, "/my/proj") == ("julia", "/my/proj")

        # ...and here is the regression, caught:
        exe, project = either_argument_bypasses("/my/julia", None)
        assert project is None, (
            "the old rule returned no project for an executable-only call -- if this ever "
            "stops being true the negative case has lost its point"
        )

        # A `--project`-less command line is what that produced, and it is exactly what the
        # driver must never be launched with.
        assert [exe, "-e", "..."] == ["/my/julia", "-e", "..."]

    def test_juliapkg_floor_covers_the_api_we_call(self):
        """
        The declared `juliapkg>=` floor admits only versions whose API this code has.

        It was `>=0.1.10`, chosen as "old enough to be safe", and 0.1.10/0.1.11/0.1.12 are a
        legal resolution of it.  In all three `resolve()` is
        `def resolve(force=False, dry_run=False)` -- so `setup_julia_environment()`, the
        command the README puts front and centre, dies with
        `TypeError: resolve() got an unexpected keyword argument 'update'` -- reproduced
        against a real `juliapkg==0.1.12` install.  `_UUID_RE` and `FileLock` are absent in
        0.1.12 too, and `find_requirements` in 0.1.10/0.1.11.  The floor was never checked
        against the floor, only against the newest two releases.
        """
        import juliapkg

        problems = juliapkg_floor_problems(
            (PACKAGE_ROOT / "pyproject.toml").read_text(), installed=juliapkg
        )
        assert problems == [], "\n".join(problems)

    def test_floor_checker_rejects_the_old_floor(self):
        """Negative case: the floor this branch shipped with must be reported."""
        assert juliapkg_floor_problems('dependencies = ["juliapkg>=0.1.22"]') == []

        problems = juliapkg_floor_problems('dependencies = ["juliapkg>=0.1.10"]')
        assert len(problems) == 1
        assert "below 0.1.22" in problems[0]
        assert "update" in problems[0]

        assert juliapkg_floor_problems('dependencies = ["ase>=3.22"]') == [
            "no `juliapkg>=X` requirement found in pyproject.toml"
        ]

        # ...and the runtime half reports a module missing the API, not just a low string.
        class Fake:
            pass

        fake = Fake()
        fake.deps = Fake()
        problems = juliapkg_floor_problems(
            'dependencies = ["juliapkg>=0.1.22"]', installed=fake
        )
        assert len(problems) == len(JULIAPKG_APIS_WE_CALL)
        assert all("does not provide" in p for p in problems)
        # every probe carries its reason into the message, so a red gate says what broke
        assert any("concurrency claim" in p for p in problems)
        assert any("headline install command" in p for p in problems)

    def test_juliapkg_discovery_descends_exactly_one_level(self, tmp_path, monkeypatch):
        """
        juliapkg finds ``<sys.path entry>/<pkg>/juliapkg.json`` and nothing deeper.

        The whole design rests on this: the declaration is found because site-packages is a
        sys.path entry and ``ase_ace`` is a subdirectory of it.  Moving the file into
        ``ase_ace/julia/`` -- which looks tidier, now that the other Julia files live there
        -- would silently orphan it, and nothing would fail loudly: juliapkg would just
        resolve an environment without ACEpotentials in it.  This pins the one-level rule as
        a fact about juliapkg rather than an assumption in our design notes.
        """
        from juliapkg.deps import deps_files

        entry = tmp_path / "site-packages"
        shallow = entry / "shallow_pkg"
        deep = entry / "deep_pkg" / "julia"
        shallow.mkdir(parents=True)
        deep.mkdir(parents=True)
        (shallow / "juliapkg.json").write_text('{"packages": {}}')
        (deep / "juliapkg.json").write_text('{"packages": {}}')

        monkeypatch.setattr(sys, "path", [str(entry)])
        found = {os.path.normpath(f) for f in deps_files()}

        assert os.path.normpath(str(shallow / "juliapkg.json")) in found
        assert os.path.normpath(str(deep / "juliapkg.json")) not in found


# ---------------------------------------------------------------------------------------
# Tier 2: build the wheel, look inside it, install it.


def wheel_problems(names):
    """
    Reasons a wheel's member list is not a usable install; empty means it is.

    `names` is the list of archive paths, as `zipfile.ZipFile.namelist()` returns them.
    """
    problems = []
    for rel in REQUIRED_ASSETS:
        want = f"ase_ace/{rel}"
        if want not in names:
            problems.append(f"{want}: not in the wheel")
    problems.extend(manifest_problems(names))
    return problems


def manifest_problems(names):
    """
    Every `Manifest.toml` in an archive, as a problem string.

    Shared by the wheel and sdist checks because the rule is the same for both, and because
    the sdist is where it was first broken: the exclusion originally lived under
    `[tool.hatch.build.targets.wheel]`, so the sdist shipped exactly what the wheel refused --
    this package's own `src/ase_ace/julia/Manifest.toml` *and* a stray `test/Manifest.toml`
    left in the build root by an accidental `julia --project=test`.  Matching on the basename
    rather than on one known path is deliberate: the stray was not at a path anyone would have
    thought to name.
    """
    return [
        f"{name}: Manifest.toml must not be shipped -- it is untracked, so it is whatever "
        f"the build machine last resolved (pyproject.toml [tool.hatch.build] `exclude`)"
        for name in names
        if Path(name).name == "Manifest.toml"
    ]


def license_problems(names, expected):
    """
    Reason an artifact's member list carries no licence text; empty means it does.

    The metadata has said ``License-Expression: MIT`` since the PEP 639 conversion, and a
    published artifact that claims a licence while shipping no licence text is exactly what a
    PyPI release must not do.  The two artifacts put the file in different places -- the wheel
    under ``<dist-info>/licenses/`` (from ``license-files``), the sdist at its root -- so the
    expected path is passed in rather than guessed.
    """
    if expected in names:
        return []
    return [
        f"{expected}: no licence text in the artifact, although the metadata declares "
        f"License-Expression: MIT (pyproject.toml `license` / `license-files`)"
    ]


def sdist_problems(names):
    """
    Reasons an sdist's member list is not a usable source release; empty means it is.

    `names` is the list of archive paths as `tarfile.getnames()` returns them, each prefixed
    with the `ase_ace-<version>/` root directory.
    """
    roots = {n.split("/")[0] for n in names if "/" in n}
    if len(roots) != 1:
        return [f"expected a single sdist root directory, got {sorted(roots)}"]
    root = roots.pop()

    problems = []
    for rel in REQUIRED_ASSETS:
        want = f"{root}/src/ase_ace/{rel}"
        if want not in names:
            problems.append(f"{want}: not in the sdist")
    problems.extend(manifest_problems(names))
    return problems


def build_wheel(dest):
    """Build the ase-ace wheel into `dest` and return its path."""
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dest),
         str(PACKAGE_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(Path(dest).glob("ase_ace-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def build_sdist(dest):
    """
    Build the ase-ace sdist into `dest` and return its path.

    Via `python -m build`, because pip has no "give me the sdist" mode -- `pip wheel` always
    goes on to build a wheel from it, which is the artifact we are trying *not* to look at
    here.  `build` is skipped rather than required, so a runner without it loses this one
    test instead of erroring.
    """
    pytest.importorskip("build", reason="python -m build is needed to produce an sdist")
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dest),
         str(PACKAGE_ROOT)],
        check=True, capture_output=True, text=True,
    )
    sdists = list(Path(dest).glob("ase_ace-*.tar.gz"))
    assert len(sdists) == 1, f"expected exactly one sdist, got {sdists}"
    return sdists[0]


class TestBuiltWheel:
    @slow
    def test_wheel_carries_the_julia_assets(self, tmp_path):
        wheel = build_wheel(tmp_path)
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
        assert wheel_problems(names) == [], "\n".join(wheel_problems(names))

    # -- negative case: the real pre-fix manifest, which this must reject ------------------

    def test_wheel_check_rejects_the_pre_fix_wheel(self):
        """
        The member list of the wheel this branch's HEAD~ produced, verbatim.  It is what a
        `pip install ase-ace` actually laid down, and it is missing every Julia asset.
        """
        pre_fix = [
            "ase_ace/__init__.py",
            "ase_ace/base.py",
            "ase_ace/calculator.py",
            "ase_ace/julia_calculator.py",
            "ase_ace/library_calculator.py",
            "ase_ace/server.py",
            "ase_ace/utils.py",
            "ase_ace/juliapkg.json",
            "ase_ace-0.1.0.dist-info/METADATA",
            "ase_ace-0.1.0.dist-info/WHEEL",
            "ase_ace-0.1.0.dist-info/RECORD",
        ]
        problems = wheel_problems(pre_fix)
        assert len(problems) == len(REQUIRED_ASSETS) - 1  # juliapkg.json it did ship
        assert all("not in the wheel" in p for p in problems)
        assert {"ase_ace/julia/ace_driver.jl", "ase_ace/julia/python_interface.jl"} == {
            p.split(":")[0] for p in problems
        }

    @slow
    def test_wheel_carries_the_license(self, tmp_path):
        """
        The wheel ships the MIT text, not just the claim of it.

        Nothing asserted this before: the tests here covered the Julia assets and the absence
        of Manifest.toml, so `export/ase-ace/LICENSE` going away would have been silent, and
        the metadata would have kept saying ``License-Expression: MIT`` over an artifact with
        no licence text -- which is what a PyPI release must not do.

        Measured while writing this: deleting ``license-files = ["LICENSE"]`` from
        pyproject.toml does *not* drop the file, because hatchling then finds ``LICENSE`` by
        its own default detection.  What this gate catches is the file itself disappearing
        (verified: both this and the sdist twin fail when it does).
        """
        wheel = build_wheel(tmp_path)
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            problems = license_problems(names, WHEEL_LICENSE)
            assert problems == [], "\n".join(problems)
            assert zf.read(WHEEL_LICENSE).decode().strip(), "the shipped LICENSE is empty"

    def test_license_check_rejects_an_artifact_without_it(self):
        """Negative case for both artifacts' licence check."""
        assert license_problems([WHEEL_LICENSE], WHEEL_LICENSE) == []
        problems = license_problems(["ase_ace/__init__.py"], WHEEL_LICENSE)
        assert len(problems) == 1
        assert "no licence text" in problems[0]

        root = "ase_ace-0.1.0"
        assert license_problems([f"{root}/LICENSE"], f"{root}/LICENSE") == []
        assert len(license_problems([f"{root}/README.md"], f"{root}/LICENSE")) == 1

    def test_wheel_check_rejects_a_shipped_manifest(self):
        good = [f"ase_ace/{rel}" for rel in REQUIRED_ASSETS]
        assert wheel_problems(good) == []
        problems = wheel_problems(good + ["ase_ace/julia/Manifest.toml"])
        assert len(problems) == 1
        assert "must not be shipped" in problems[0]


class TestBuiltSdist:
    """
    The sdist is the second artifact, and it needs its own gate.

    A `[tool.hatch.build.targets.wheel] exclude` says nothing about the sdist, so for one
    commit this package shipped a wheel with no Manifest and an sdist with two of them. Anyone
    unpacking that sdist and following README section 2 would have instantiated the build
    machine's resolve rather than their own.
    """

    @slow
    def test_sdist_carries_the_julia_assets_and_no_manifest(self, tmp_path):
        sdist = build_sdist(tmp_path)
        with tarfile.open(sdist) as tf:
            names = tf.getnames()
        assert sdist_problems(names) == [], "\n".join(sdist_problems(names))

    # -- negative case: the real pre-fix sdist, which this must reject --------------------

    @slow
    def test_sdist_carries_the_license(self, tmp_path):
        sdist = build_sdist(tmp_path)
        with tarfile.open(sdist) as tf:
            names = tf.getnames()
        roots = {n.split("/")[0] for n in names if "/" in n}
        assert len(roots) == 1, sorted(roots)
        expected = f"{roots.pop()}/LICENSE"
        problems = license_problems(names, expected)
        assert problems == [], "\n".join(problems)

    def test_sdist_check_rejects_the_wheel_only_exclusion(self):
        """
        The member list actually produced while `exclude` sat under the wheel target, trimmed
        to what matters.  Both Manifests must be reported -- the package's own, and the stray
        from a `test/` directory that had no business existing in the build root.
        """
        root = "ase_ace-0.1.0"
        names = [
            f"{root}/pyproject.toml",
            f"{root}/README.md",
            f"{root}/src/ase_ace/__init__.py",
            f"{root}/src/ase_ace/juliapkg.json",
            f"{root}/src/ase_ace/julia/Manifest.toml",
            f"{root}/src/ase_ace/julia/Project.toml",
            f"{root}/src/ase_ace/julia/ace_driver.jl",
            f"{root}/src/ase_ace/julia/python_interface.jl",
            f"{root}/test/Manifest.toml",
            f"{root}/tests/conftest.py",
        ]
        problems = sdist_problems(names)
        assert len(problems) == 2
        assert all("must not be shipped" in p for p in problems)
        assert any(p.startswith(f"{root}/test/Manifest.toml") for p in problems), (
            "the stray build-root Manifest must be caught too -- it is not at a path anyone "
            "would have named in advance, which is why the check matches on the basename"
        )

    def test_sdist_check_rejects_a_missing_asset(self):
        root = "ase_ace-0.1.0"
        good = [f"{root}/src/ase_ace/{rel}" for rel in REQUIRED_ASSETS]
        assert sdist_problems(good) == []
        problems = sdist_problems([n for n in good if not n.endswith("ace_driver.jl")])
        assert len(problems) == 1
        assert "not in the sdist" in problems[0]


class TestNonEditableInstall:
    """
    The acceptance test in full: a fresh venv, a non-editable install, assets resolved from
    site-packages by a subprocess whose working directory is nowhere near this checkout.
    """

    def _probe_installed_wheel(self, tmp_path):
        """
        Build the wheel, install it into a throwaway venv, and run the probe inside it.

        Returns the probe's decoded JSON.  Shared by the two tests below because they ask
        different questions of the same install and building twice is pure cost.
        """
        wheel = build_wheel(tmp_path / "wheel")

        venv = tmp_path / "venv"
        # `--system-site-packages` so `ase` and `numpy` come from the host: this test is about
        # where ase-ace's own files land, and downloading its dependencies would make it fail
        # for network reasons rather than packaging ones.
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
                       check=True, capture_output=True, text=True)
        python = venv / "bin" / "python"
        if not python.exists():  # Windows layout
            python = venv / "Scripts" / "python.exe"

        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-q", str(wheel)],
                       check=True, capture_output=True, text=True)

        probe = r"""
import json  # gate-A-negative: this block is Python for a subprocess, not Julia
from pathlib import Path
import ase_ace  # gate-A-negative: this block is Python for a subprocess, not Julia
from ase_ace.server import get_julia_assets_path
from ase_ace.julia_calculator import _INTERFACE_PATH
from juliapkg.deps import deps_files, find_requirements

pkg = Path(ase_ace.__file__).resolve().parent
assets = get_julia_assets_path().resolve()
iface = _INTERFACE_PATH.resolve()
want = str((pkg / "juliapkg.json").resolve())
files = [str(Path(f).resolve()) for f in deps_files()]
compat, specs = find_requirements()
print(json.dumps({
    "package_dir": str(pkg),
    "assets": str(assets),
    "assets_exists": assets.is_dir(),
    "driver_exists": (assets / "ace_driver.jl").is_file(),
    "interface": str(iface),
    "interface_exists": iface.is_file(),
    "juliapkg_exists": (pkg / "juliapkg.json").is_file(),
    "project_toml_shipped": (assets / "Project.toml").exists(),
    "manifest_shipped": (assets / "Manifest.toml").exists(),
    "juliapkg_json_discovered": want in files,
    "deps_files": files,
    "declared": sorted(spec.name for spec in specs),
    "julia_compat": None if compat is None else str(compat),
}))
"""
        # cwd well away from the checkout, so nothing is found by accident
        result = subprocess.run([str(python), "-c", probe], cwd=str(tmp_path),
                                capture_output=True, text=True)
        assert result.returncode == 0, (
            f"probe failed in the installed venv:\n{result.stdout}\n{result.stderr}"
        )
        info = json.loads(result.stdout.strip().splitlines()[-1])

        assert "site-packages" in info["package_dir"], (
            f"expected a non-editable install; got {info['package_dir']}"
        )
        # ...and specifically THIS venv's site-packages.  The venv is created with
        # --system-site-packages, and a developer machine may well have another ase_ace
        # installed there (this one does); if that copy were the one imported, every
        # assertion below would be about it rather than about the wheel just built.
        assert info["package_dir"].startswith(str(venv.resolve())), (
            f"the probe imported {info['package_dir']}, which is not inside the venv at "
            f"{venv}"
        )
        return info

    @slow
    def test_assets_resolve_from_a_non_editable_install(self, tmp_path):
        info = self._probe_installed_wheel(tmp_path)

        assert info["assets_exists"], f"Julia assets missing: {info['assets']}"
        assert info["driver_exists"], "ace_driver.jl missing from the installed package"
        assert info["interface_exists"], f"python_interface.jl missing: {info['interface']}"
        assert info["juliapkg_exists"], "juliapkg.json missing from the installed package"
        assert not info["manifest_shipped"], (
            "Manifest.toml was shipped; pyproject.toml excludes it on purpose"
        )
        assert not info["project_toml_shipped"], (
            "julia/Project.toml is back in the artifact -- that is the second dependency "
            "declaration this package folded onto juliapkg to get rid of"
        )
        assert info["assets"].startswith(info["package_dir"]), (
            "the Julia scripts must live inside the installed package, not beside it"
        )

    @slow
    def test_juliapkg_discovers_the_installed_declaration(self, tmp_path):
        """
        juliapkg finds the shipped juliapkg.json from site-packages, and it is complete.

        This is the load-bearing invariant of the fold: the socket backend no longer carries
        its own project, so if juliapkg does not find this file the Julia environment is
        built without ACEpotentials in it -- and nothing fails until a driver subprocess dies
        with `Package ACEpotentials not found`.  ``deps_files()`` descends exactly one level
        into each sys.path entry (pinned in tier 1), so this passes only while the file sits
        directly inside the package.

        No Julia and no network: ``deps_files()`` and ``find_requirements()`` only read files.
        """
        info = self._probe_installed_wheel(tmp_path)

        assert info["juliapkg_json_discovered"], (
            "juliapkg did not find the installed juliapkg.json.  It looked in:\n"
            + "\n".join(info["deps_files"])
        )
        # A superset, not equality: the venv is created with --system-site-packages, so
        # juliacall's own juliapkg.json (PythonCall, OpenSSL_jll) may be merged in too.
        assert EXPECTED_JULIA_PACKAGES <= set(info["declared"]), (
            f"missing from the merged requirements: "
            f"{sorted(EXPECTED_JULIA_PACKAGES - set(info['declared']))}"
        )
        assert info["julia_compat"], "no julia version constraint reached juliapkg"
