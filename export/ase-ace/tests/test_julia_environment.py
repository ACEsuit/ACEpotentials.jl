"""
The declared Julia dependency set is SUFFICIENT, not merely self-consistent.

WHY THIS FILE EXISTS.

``tests/test_packaging.py`` gates the *declaration*: that there is exactly one of it, that
every ``using`` in the repository's Julia-driving code appears in it, that its UUIDs parse,
that juliapkg can find it from site-packages.  Every one of those is a statement about text.

None of them says the declared set can actually **run** anything.  That gap cost a red CI:
``ACEfit`` was dropped from ``juliapkg.json`` (correctly -- nothing shipped needs it), the
declaration stayed internally consistent, every gate passed, and the fixture-fitting step
died with ``ArgumentError: Package ACEfit not found in current path``.

It passed locally for the worst possible reason.  juliapkg does not resolve *our*
``juliapkg.json`` -- it resolves the **merge** of every ``juliapkg.json`` on ``sys.path``, and
this developer's machine has a stale second ``ase_ace`` in ``~/.local`` whose copy still
declares ``ACEfit``.  Measured:

    merged packages: ['ACEfit', 'ACEpotentials', 'ArgParse', 'AtomsBase', 'AtomsCalculators',
                      'IPICalculator', 'OpenSSL_jll', 'PythonCall', 'StaticArrays',
                      'Unitful', 'UnitfulAtomic']
    julia compat: =1.11.5

Eleven packages, not eight, and a Julia pin from a release two versions old.  Every local run
was against an environment the declaration does not describe.

So "clean" here has to mean clean in **three** senses, and a first draft of this file got only
the first two:

1. a fresh ``PYTHON_JULIAPKG_PROJECT``, so no previously-resolved project is reused;
2. a throwaway virtualenv built WITHOUT system or user site-packages, so no other
   installation's ``juliapkg.json`` is on ``sys.path``;
3. only a copy of *this* declaration placed on the path -- and then an assertion that what
   juliapkg merged is exactly what we declared, so cleanliness is checked rather than hoped
   for.  Without (3) the first draft of this file passed its first two tests and failed its
   third, correctly reporting that the other two had proved nothing.

``JULIA_DEPOT_PATH`` is deliberately *not* overridden: the depot is a download cache, not a
declaration.  What decides whether a package is loadable is the project, and this project is
built from ``juliapkg.json`` alone.

COST.  It creates a venv, resolves a Julia environment and fits a small model: minutes warm,
tens of minutes and a large download cold.  It is therefore gated on ``ACE_TEST_JULIA=1``,
separately from ``ACE_TEST_PACKAGING`` -- which stays fast, and stays skip-free -- and it
lives in its own file so the packaging gate's "nothing skipped" property is unaffected.  CI
sets it in the ``ase-ace (julia calculator)`` job: the job that went red, and the one that
already caches ``~/.julia``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DECLARATION = PACKAGE_ROOT / "src" / "ase_ace" / "juliapkg.json"

RUN_JULIA = os.environ.get("ACE_TEST_JULIA") == "1"
needs_julia = pytest.mark.skipif(
    not RUN_JULIA,
    reason="set ACE_TEST_JULIA=1 to resolve a clean Julia environment and run the "
    "fixture-fitting path in it (minutes warm, much longer cold, needs the network)",
)


def declared_packages():
    with open(DECLARATION) as f:
        return sorted(json.load(f).get("packages", {}))


class CleanJuliaEnv:
    """A Julia environment built from this package's declaration and nothing else."""

    def __init__(self, python, env, project):
        self.python = python
        self.env = env
        self.project = project
        self.executable = None

    def julia(self, source, timeout=3600):
        """Run `source` in the clean environment; returns the CompletedProcess."""
        return subprocess.run(
            [self.executable, f"--project={self.project}", "-e", source],
            cwd=str(self.project), env=self.env, capture_output=True, text=True,
            timeout=timeout,
        )


@pytest.fixture(scope="session")
def clean_julia_env(tmp_path_factory):
    """
    Build the environment once per session -- three resolves would be three times the wait.
    """
    if not RUN_JULIA:
        pytest.skip("ACE_TEST_JULIA=1 not set")

    tmp = tmp_path_factory.mktemp("clean_julia")

    # (2) a venv with NO system and NO user site-packages.  `python -m venv` without
    # --system-site-packages gets us the first; PYTHONNOUSERSITE gets us the second, which is
    # the one that matters here -- the stale ase_ace that hid the CI failure lives in
    # ~/.local, i.e. user site.
    venv = tmp / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)],
                   check=True, capture_output=True, text=True)
    python = venv / "bin" / "python"
    if not python.exists():  # Windows layout
        python = venv / "Scripts" / "python.exe"
    subprocess.run([str(python), "-m", "pip", "install", "-q", "juliapkg>=0.1.22"],
                   check=True, capture_output=True, text=True, timeout=600)

    # (3) exactly one declaration on the path: a copy of ours.  juliapkg's discovery descends
    # one level into each sys.path entry, so `<pathdir>/<pkg>/juliapkg.json` is found -- the
    # same shape as site-packages/ase_ace/juliapkg.json.
    decl_root = tmp / "declaration"
    (decl_root / "ase_ace_under_test").mkdir(parents=True)
    (decl_root / "ase_ace_under_test" / "juliapkg.json").write_text(DECLARATION.read_text())

    # (1) a project directory made for this run.
    project = tmp / "julia_env"
    project.mkdir()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(decl_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHON_JULIAPKG_PROJECT"] = str(project)
    env.pop("PYTHON_JULIAPKG_OFFLINE", None)
    env.pop("PYTHON_JULIAPKG_EXE", None)

    handle = CleanJuliaEnv(str(python), env, str(project))

    # Check cleanliness BEFORE resolving, and make it an assertion rather than an assumption:
    # what juliapkg is about to resolve must be exactly what this package declares.
    probe = (
        "import json;"
        "from juliapkg.deps import deps_files, find_requirements;"
        "c, specs = find_requirements();"
        "print(json.dumps({'files': sorted(deps_files()),"
        " 'merged': sorted(s.name for s in specs),"
        " 'compat': None if c is None else str(c)}))"
    )
    seen = subprocess.run([str(python), "-c", probe], cwd=str(tmp), env=env,
                          capture_output=True, text=True, timeout=300)
    assert seen.returncode == 0, f"{seen.stdout}\n{seen.stderr}"
    info = json.loads(seen.stdout.strip().splitlines()[-1])
    assert info["merged"] == declared_packages(), (
        "the environment under test is NOT clean: juliapkg merged "
        f"{info['merged']}, but this package declares {declared_packages()}.\n"
        "juliapkg resolves the merge of every juliapkg.json on sys.path, so a second "
        "ase_ace installation (or juliacall) leaking in makes every assertion below "
        "meaningless -- that is exactly how the ACEfit regression passed locally and "
        "failed in CI.\n"
        f"declarations found: {info['files']}"
    )

    locate = ("import juliapkg, json;"
              "print(json.dumps({'exe': juliapkg.executable(),"
              " 'project': juliapkg.project()}))")
    located = subprocess.run([str(python), "-c", locate], cwd=str(tmp), env=env,
                             capture_output=True, text=True, timeout=3600)
    assert located.returncode == 0, (
        f"juliapkg could not resolve the declaration into a clean project:\n"
        f"{located.stdout}\n{located.stderr}"
    )
    resolved = json.loads(located.stdout.strip().splitlines()[-1])
    assert resolved["project"] == str(project)
    handle.executable = resolved["exe"]
    return handle


class TestDeclaredSetIsSufficient:
    @needs_julia
    def test_every_declared_package_loads(self, clean_julia_env):
        """The names in juliapkg.json resolve together and all load."""
        names = declared_packages()
        result = clean_julia_env.julia(
            f"using {', '.join(names)}; println(\"ALL_LOAD_OK \", VERSION)"
        )
        assert result.returncode == 0, (
            f"the declared set does not load:\n{result.stdout}\n{result.stderr}"
        )
        assert "ALL_LOAD_OK" in result.stdout

    @needs_julia
    def test_the_fixture_fitting_path_runs(self, clean_julia_env):
        """
        The Julia that `conftest.create_test_model` and the CI workflow actually run.

        This is the assertion that was missing.  It fits the same tiny Si model with the same
        solver expression, in an environment built from the declaration alone -- so a package
        the tooling needs but the declaration omits fails HERE, in a test, rather than in the
        CI job that produces the fixture every other Julia test depends on.
        """
        source = """
        using ACEpotentials
        using Random
        Random.seed!(12345)

        # `ACEpotentials.ACEfit`, exactly as tests/conftest.py and the CI workflow write it.
        # ACEfit is NOT declared; this reaches it through ACEpotentials' own `using ACEfit`
        # binding.  If that ever stops working, this line is where it is caught.
        solver = ACEpotentials.ACEfit.BLR()

        model = ACEpotentials.ace1_model(
            elements = [:Si], order = 2, totaldegree = 6, rcut = 5.5)
        dataset = ACEpotentials.example_dataset("Si_tiny")
        ACEpotentials.acefit!(dataset.train, model;
            energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial",
            weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0)),
            solver = solver)
        mktempdir() do d
            ACEpotentials.save_model(model, joinpath(d, "m.json"))
            println("FIXTURE_PATH_OK ", isfile(joinpath(d, "m.json")))
        end
        """
        result = clean_julia_env.julia(source)
        assert result.returncode == 0, (
            f"the fixture-fitting path does not run in a clean declared environment:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert "FIXTURE_PATH_OK true" in result.stdout

    @needs_julia
    def test_undeclared_packages_really_are_absent(self, clean_julia_env):
        """
        The clean environment genuinely lacks what the declaration does not name.

        Belt to the fixture's braces: the fixture asserts the *declaration* juliapkg merged is
        ours alone, this asserts the *resolved project* behaves accordingly.  `ACEfit` is the
        case that bit -- reachable as `ACEpotentials.ACEfit`, but it must not be loadable as a
        package of its own, because it is not declared.

        If a future change legitimately re-declares ACEfit this test fails, by design: the
        declaration, this file, the packaging gate's EXPECTED_JULIA_PACKAGES, the README and
        the report all have to move together.
        """
        assert "ACEfit" not in declared_packages(), (
            "ACEfit is declared now -- update this test, EXPECTED_JULIA_PACKAGES, the README "
            "and the report's 'drops ACEfit' narrative together"
        )
        result = clean_julia_env.julia('using ACEfit; println("LOADED")')
        assert result.returncode != 0, (
            "ACEfit loaded as a package in an environment that does not declare it -- the "
            "environment under test is contaminated, so the tests above prove nothing"
        )
        assert "ACEfit not found" in result.stderr, result.stderr

    @needs_julia
    def test_the_qualified_solver_expression_is_the_one_that_works(self, clean_julia_env):
        """
        Pin the three ways of reaching ACEfit, since only two of them work.

        Measured in exactly this environment, and the reason the fix is
        `ACEpotentials.ACEfit.BLR()` rather than either alternative:

            using ACEfit                        -> ArgumentError: Package ACEfit not found
            using ACEpotentials; BLR()          -> UndefVarError: `BLR` not defined
            using ACEpotentials; ACEfit.BLR()   -> works, but only because Reexport
                                                   re-exports the MODULE NAME
            using ACEpotentials; ACEpotentials.ACEfit.BLR()  -> works via the `using` binding

        The third is what the code used before this fix.  It is fine today, but it depends on
        a re-export whose own author wrote "should we re-export ACEfit? I'm not convinced"
        (src/ACEpotentials.jl:8); the fourth does not depend on it at all.
        """
        bare = clean_julia_env.julia("using ACEpotentials; BLR()")
        assert bare.returncode != 0, (
            "bare BLR() works now -- ACEfit has started exporting it, so the note above and "
            "in tests/conftest.py should be updated"
        )
        assert "BLR" in bare.stderr

        qualified = clean_julia_env.julia(
            "using ACEpotentials; println(typeof(ACEpotentials.ACEfit.BLR()))"
        )
        assert qualified.returncode == 0, (
            "ACEpotentials.ACEfit.BLR() no longer resolves -- the fixture path and the CI "
            f"workflow both use it:\n{qualified.stdout}\n{qualified.stderr}"
        )
        assert "BLR" in qualified.stdout
