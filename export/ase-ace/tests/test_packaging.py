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
SEVAL_RE = re.compile(r"""seval\(\s*['"]\s*((?:using|import)\s[^'"]+)['"]""")


def julia_packages_used(text, kind="julia"):
    """
    Top-level Julia package names that a chunk of source ``using``s or ``import``s.

    Handles the three forms that appear in the shipped files:
    ``using A, B``, ``using A: x, y`` (the package is ``A``) and ``using A.Sub: x``
    (likewise ``A``).  Relative forms (``using .Mod``) are dropped -- they name a module
    defined in the same file, not a dependency.

    With ``kind="python"`` it instead reads `jl.seval("using X")` out of Python source, and
    ONLY that.  The two are separate because `USING_RE` is anchored to the start of a line
    and would otherwise swallow Python's own `import os` / `import numpy as np`.  Without the
    python mode this checker's name overpromised: it read .jl files only, while five of the
    packages the juliacall backend loads are named in Python string literals in
    julia_calculator.py.  Nothing had drifted, but the invariant was narrower than its name.
    """
    if kind == "python":
        clauses = [m.split(None, 1)[1] for m in SEVAL_RE.findall(text)]
    elif kind == "julia":
        clauses = USING_RE.findall(text)
    else:
        raise ValueError(kind)

    found = set()
    for clause in clauses:
        clause = clause.split(":", 1)[0]
        for name in clause.split(","):
            name = name.strip()
            if not name or name.startswith("."):
                continue
            found.add(name.split(".", 1)[0])
    return found


def undeclared_packages(sources, declared, python_sources=()):
    """
    Package names loaded by the sources but absent from `declared`; empty means consistent.

    `sources` is an iterable of Julia source strings and `python_sources` of Python source
    strings (scanned for `jl.seval("using X")` only); `declared` is the set of names in
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
JULIAPKG_APIS_WE_CALL = {
    # utils.setup_julia_environment() passes update=; resolve() has no such parameter up to
    # and including 0.1.12, so the README's headline install command raises TypeError there.
    "resolve(update=)": lambda jp: "update" in __import__("inspect").signature(
        jp.deps.resolve
    ).parameters,
    "executable": lambda jp: callable(jp.executable),
    "project": lambda jp: callable(jp.project),
    # used by the tier-2 probe and by test_juliapkg_json_is_well_formed
    "deps.find_requirements": lambda jp: callable(jp.deps.find_requirements),
    "deps._UUID_RE": lambda jp: jp.deps._UUID_RE is not None,
    # the cross-process lock the fold claims to inherit; absent in 0.1.12 and earlier
    "deps.FileLock": lambda jp: jp.deps.FileLock is not None,
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
        for name, probe in sorted(JULIAPKG_APIS_WE_CALL.items()):
            try:
                ok = probe(installed)
            except Exception as e:  # AttributeError on an old version
                ok = False
                name = f"{name} ({type(e).__name__})"
            if not ok:
                problems.append(
                    f"installed juliapkg does not provide {name}, which ase-ace calls"
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
        Every package the shipped .jl files load is in juliapkg.json.

        This is the invariant the two declarations broke: ``ace_driver.jl`` does
        ``using ArgParse``, ``using IPICalculator`` and ``using UnitfulAtomic``, and for the
        package's whole life none of the three was in juliapkg.json -- so the socket backend
        was unusable in any environment juliapkg had built.
        """
        import ase_ace

        package_dir = Path(ase_ace.__file__).resolve().parent
        declared = json.loads((package_dir / "juliapkg.json").read_text())["packages"]
        sources = [
            (package_dir / "julia" / name).read_text()
            for name in ("ace_driver.jl", "python_interface.jl")
        ]
        # ...and the Python file that loads Julia packages by `seval`.
        python_sources = [(package_dir / "julia_calculator.py").read_text()]
        assert undeclared_packages(sources, declared, python_sources) == []

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
        assert julia_packages_used(
            "        jl.seval('using StaticArrays')\n", kind="python"
        ) == {"StaticArrays"}
        assert undeclared_packages(
            [], declared, python_sources=["jl.seval('using Nowhere')\n"]
        ) == ["Nowhere"]
        # ...without mistaking Python's own imports for Julia dependencies, which is why the
        # two modes are separate rather than one regex over everything.
        assert julia_packages_used(
            "import os\nimport numpy as np\nfrom pathlib import Path\n", kind="python"
        ) == set()

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
import json
from pathlib import Path
import ase_ace
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
