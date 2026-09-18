"""
The Julia assets resolve from the INSTALLED package, in a wheel as well as in a checkout.

WHY THIS FILE EXISTS.

`ase_ace.server.get_julia_project_path()` and `ase_ace.julia_calculator._INTERFACE_PATH` used
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
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# The Julia assets that must travel with the installed package.  ``Manifest.toml`` is
# deliberately absent: see the exclusion comment in pyproject.toml.
REQUIRED_ASSETS = (
    "julia/Project.toml",
    "julia/ace_driver.jl",
    "julia/python_interface.jl",
    "juliapkg.json",
)

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

    def test_julia_project_path_is_inside_the_package(self):
        import ase_ace
        from ase_ace.server import get_julia_project_path

        package_dir = Path(ase_ace.__file__).resolve().parent
        project = get_julia_project_path().resolve()

        # `relative_to` raises if `project` is not under the package -- which is exactly what
        # the pre-fix `parent.parent.parent / "julia"` did, editable install or not.
        project.relative_to(package_dir)
        assert (project / "Project.toml").exists()
        assert (project / "ace_driver.jl").exists()

    def test_interface_path_is_inside_the_package(self):
        pytest.importorskip("numpy")
        pytest.importorskip("ase")

        import ase_ace
        from ase_ace.julia_calculator import _INTERFACE_PATH

        package_dir = Path(ase_ace.__file__).resolve().parent
        interface = _INTERFACE_PATH.resolve()

        interface.relative_to(package_dir)
        assert interface.exists()

    def test_utils_default_project_agrees(self):
        """`setup_julia_environment` defaults through the same one function, not a copy."""
        from ase_ace.server import get_julia_project_path
        from ase_ace import utils

        src = Path(utils.__file__).read_text()
        assert "get_julia_project_path" in src, (
            "utils.py must route its default Julia project through server."
            "get_julia_project_path(); a second copy of the layout assumption is how this "
            "bug got two homes in the first place"
        )
        assert get_julia_project_path().is_dir()

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
        assert len(problems) == 3
        assert all("not in the wheel" in p for p in problems)

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

    @slow
    def test_assets_resolve_from_a_non_editable_install(self, tmp_path):
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
from ase_ace.server import get_julia_project_path
from ase_ace.julia_calculator import _INTERFACE_PATH

pkg = Path(ase_ace.__file__).resolve().parent
project = get_julia_project_path().resolve()
iface = _INTERFACE_PATH.resolve()
print(json.dumps({
    "package_dir": str(pkg),
    "project": str(project),
    "project_exists": project.is_dir(),
    "driver_exists": (project / "ace_driver.jl").is_file(),
    "interface": str(iface),
    "interface_exists": iface.is_file(),
    "juliapkg_exists": (pkg / "juliapkg.json").is_file(),
    "manifest_shipped": (project / "Manifest.toml").exists(),
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
        assert info["project_exists"], f"Julia project missing: {info['project']}"
        assert info["driver_exists"], "ace_driver.jl missing from the installed project"
        assert info["interface_exists"], f"python_interface.jl missing: {info['interface']}"
        assert info["juliapkg_exists"], "juliapkg.json missing from the installed package"
        assert not info["manifest_shipped"], (
            "Manifest.toml was shipped; pyproject.toml excludes it on purpose"
        )
        assert info["project"].startswith(info["package_dir"]), (
            "the Julia project must live inside the installed package, not beside it"
        )
