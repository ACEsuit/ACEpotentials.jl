#!/usr/bin/env python3
"""
Benchmark comparison of ACE potential interfaces.

Compares:
1. Native Julia (via subprocess)
2. Python ACECalculator (socket-based, multi-threaded)
3. Python ACELibraryCalculator (compiled library, single-threaded)
4. LAMMPS with ACE plugin (MPI + OpenMP)

Usage:
    python benchmark_interfaces.py --library path/to/libace.so [--model path/to/model.json] \
        [--lammps /path/to/lmp] [--plugin /path/to/aceplugin.so]
"""

import os
import sys
import time
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write as ase_write


def create_structures(sizes, species='Si'):
    """Create supercells of various sizes for given species."""
    from ase.data import chemical_symbols

    structures = {}

    # Define base structures for common elements
    if species == 'Si':
        base = bulk('Si', 'diamond', a=5.43)
    elif species == 'Ti':
        base = bulk('Ti', 'hcp', a=2.95)
    elif species == 'Al':
        base = bulk('Al', 'fcc', a=4.05)
    else:
        # Default to FCC with estimated lattice constant
        base = bulk(species, 'fcc', a=4.0)

    for n in sizes:
        atoms = base * (n, n, n)
        # Add small perturbation
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
        structures[len(atoms)] = atoms

    return structures


def benchmark_library_calculator(library_path, structures, n_iterations=5):
    """Benchmark ACELibraryCalculator."""
    try:
        from ase_ace import ACELibraryCalculator
    except ImportError:
        return None

    calc = ACELibraryCalculator(library_path)
    results = {}

    for natoms, atoms in structures.items():
        atoms = atoms.copy()
        atoms.calc = calc

        # Warmup
        atoms.get_potential_energy()
        atoms.get_forces()

        # Timed iterations - perturb slightly each time to invalidate cache
        times = []
        for i in range(n_iterations):
            # Small perturbation to avoid ASE caching
            atoms.positions[0, 0] += 0.0001

            start = time.perf_counter()
            atoms.get_potential_energy()
            atoms.get_forces()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results[natoms] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
        }

    return results


def benchmark_socket_calculator(model_path, structures, num_threads, n_iterations=5, timeout=180):
    """Benchmark ACECalculator (socket-based)."""
    try:
        from ase_ace import ACECalculator
    except ImportError:
        return None

    julia_project = Path(__file__).parent / "julia"

    results = {}

    try:
        with ACECalculator(
            model_path,
            num_threads=num_threads,
            timeout=timeout,
            julia_project=str(julia_project),
        ) as calc:
            for natoms, atoms in structures.items():
                atoms = atoms.copy()
                atoms.calc = calc

                # Warmup (includes JIT on first call)
                atoms.get_potential_energy()
                atoms.get_forces()

                # Timed iterations
                times = []
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    atoms.get_potential_energy()
                    atoms.get_forces()
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                results[natoms] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                }
    except Exception as e:
        print(f"Socket calculator failed: {e}")
        return None

    return results


def benchmark_native_julia(model_path, structures, num_threads, n_iterations=5):
    """Benchmark native Julia evaluation."""

    # Create a Julia benchmark script
    julia_script = '''
    using ACEpotentials
    using AtomsBase
    using Unitful
    using UnitfulAtomic
    using JSON
    using Statistics

    function create_system(positions, cell, species)
        n = size(positions, 2)
        atoms = [AtomsBase.Atom(Symbol(species[i]), positions[:, i] .* u"Å") for i in 1:n]
        cell_matrix = cell .* u"Å"
        return periodic_system(atoms, cell_matrix)
    end

    function benchmark_model(model_path, structures_json, n_iterations)
        # Load model
        potential, meta = ACEpotentials.load_model(model_path)

        structures = JSON.parse(structures_json)
        results = Dict{Int, Dict{String, Float64}}()

        for (natoms_str, data) in structures
            natoms = parse(Int, natoms_str)
            positions = hcat(data["positions"]...)
            cell = hcat(data["cell"]...)
            species = data["species"]

            system = create_system(positions, cell, species)

            # Warmup
            AtomsCalculators.potential_energy(system, potential)
            AtomsCalculators.forces(system, potential)

            # Timed iterations
            times = Float64[]
            for _ in 1:n_iterations
                t0 = time()
                AtomsCalculators.potential_energy(system, potential)
                AtomsCalculators.forces(system, potential)
                push!(times, time() - t0)
            end

            results[natoms] = Dict(
                "mean" => mean(times),
                "std" => std(times),
                "min" => minimum(times),
            )
        end

        return results
    end

    # Parse arguments
    model_path = ARGS[1]
    structures_json = ARGS[2]
    n_iterations = parse(Int, ARGS[3])

    results = benchmark_model(model_path, structures_json, n_iterations)
    println(JSON.json(results))
    '''

    # Prepare structures as JSON
    structures_data = {}
    for natoms, atoms in structures.items():
        structures_data[str(natoms)] = {
            'positions': atoms.positions.T.tolist(),
            'cell': atoms.cell[:].tolist(),
            'species': [str(s) for s in atoms.symbols],
        }

    structures_json = json.dumps(structures_data)

    # Run Julia benchmark
    julia_project = Path(__file__).parent / "julia"

    env = os.environ.copy()
    env['JULIA_NUM_THREADS'] = str(num_threads)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        script_path = f.name

    try:
        result = subprocess.run(
            ['julia', f'--project={julia_project}', script_path,
             str(model_path), structures_json, str(n_iterations)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"Julia benchmark failed:\n{result.stderr}")
            return None

        # Parse JSON output from last line
        output_lines = result.stdout.strip().split('\n')
        results_json = output_lines[-1]
        results_raw = json.loads(results_json)

        # Convert keys to int
        results = {int(k): v for k, v in results_raw.items()}
        return results

    except subprocess.TimeoutExpired:
        print("Julia benchmark timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse Julia output: {e}")
        print(f"Output was: {result.stdout}")
        return None
    finally:
        os.unlink(script_path)


def benchmark_lammps(library_path, plugin_path, lammps_exe, structures, n_iterations=5, omp_threads=1):
    """Benchmark LAMMPS with ACE plugin."""

    if not Path(lammps_exe).exists() and subprocess.run(['which', lammps_exe], capture_output=True).returncode != 0:
        print(f"LAMMPS executable not found: {lammps_exe}")
        return None

    if not Path(plugin_path).exists():
        print(f"ACE plugin not found: {plugin_path}")
        return None

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for natoms, atoms in structures.items():
            # Write structure to LAMMPS data file
            data_file = tmpdir / f"structure_{natoms}.data"
            ase_write(str(data_file), atoms, format='lammps-data')

            # Get unique species
            species = list(set(atoms.symbols))
            species_str = " ".join(species)

            # Get masses for each species
            from ase.data import atomic_masses, atomic_numbers
            mass_lines = []
            for i, s in enumerate(species, 1):
                z = atomic_numbers[s]
                mass = atomic_masses[z]
                mass_lines.append(f"mass {i} {mass}")
            mass_str = "\n".join(mass_lines)

            # Create LAMMPS input script for benchmarking
            # Use velocity verlet with 0 timestep to force evaluation without moving atoms
            input_script = f"""
# ACE potential benchmark
units metal
atom_style atomic
boundary p p p

read_data {data_file}

# Set masses
{mass_str}

# Load ACE plugin and potential
plugin load {plugin_path}
pair_style ace
pair_coeff * * {library_path} {species_str}

# Use NVE with dt=0 to get force evaluation timing
fix 1 all nve
timestep 0.0

# Warmup - run a few steps
thermo 10
run 10

# Reset timer for actual benchmark
timer timeout 0 every 1
reset_timestep 0

# Benchmark run
thermo {n_iterations}
run {n_iterations}
"""

            input_file = tmpdir / f"in.benchmark_{natoms}"
            with open(input_file, 'w') as f:
                f.write(input_script)

            # Run LAMMPS
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(omp_threads)

            try:
                start = time.perf_counter()
                result = subprocess.run(
                    [lammps_exe, '-in', str(input_file)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300,
                    cwd=tmpdir,
                )
                total_time = time.perf_counter() - start

                if result.returncode != 0:
                    print(f"LAMMPS failed for {natoms} atoms:")
                    print(result.stderr[:500])
                    continue

                # Parse timing from LAMMPS output
                # Look for the last "Loop time" line (from benchmark run)
                loop_time = None
                n_steps = None
                for line in result.stdout.split('\n'):
                    if 'Loop time of' in line:
                        parts = line.split()
                        try:
                            loop_time = float(parts[3])
                            # Extract number of steps from "on X procs for Y steps"
                            for_idx = parts.index('for')
                            n_steps = int(parts[for_idx + 1])
                        except (IndexError, ValueError):
                            pass

                if loop_time is not None and n_steps is not None and n_steps > 0:
                    time_per_step = loop_time / n_steps
                    results[natoms] = {
                        'mean': time_per_step,
                        'std': 0,  # Single run, no std
                        'min': time_per_step,
                    }
                else:
                    print(f"Could not parse LAMMPS timing for {natoms} atoms")
                    print(f"Last few lines: {result.stdout.split(chr(10))[-10:]}")

            except subprocess.TimeoutExpired:
                print(f"LAMMPS timed out for {natoms} atoms")
                continue

    return results if results else None


def print_results(results, title):
    """Print benchmark results table."""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'Atoms':>8} {'Mean (ms)':>12} {'Std (ms)':>10} {'Min (ms)':>10}")
    print("-" * 44)

    for natoms in sorted(results.keys()):
        r = results[natoms]
        print(f"{natoms:>8} {r['mean']*1000:>12.2f} {r['std']*1000:>10.2f} {r['min']*1000:>10.2f}")


def print_comparison(all_results, sizes):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON: Time per evaluation (energy + forces) in milliseconds")
    print("=" * 80)

    headers = ["Atoms"] + list(all_results.keys())
    print(f"{'Atoms':>8}", end="")
    for name in all_results.keys():
        print(f"{name:>18}", end="")
    print()
    print("-" * (8 + 18 * len(all_results)))

    for natoms in sorted(sizes):
        print(f"{natoms:>8}", end="")
        for name, results in all_results.items():
            if results and natoms in results:
                print(f"{results[natoms]['mean']*1000:>18.2f}", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()

    # Print speedup relative to library calculator
    if "Library (1T)" in all_results and all_results["Library (1T)"]:
        print("\nSpeedup relative to Library (1T):")
        print("-" * (8 + 18 * len(all_results)))
        base = all_results["Library (1T)"]

        for natoms in sorted(sizes):
            if natoms not in base:
                continue
            print(f"{natoms:>8}", end="")
            base_time = base[natoms]['mean']
            for name, results in all_results.items():
                if results and natoms in results:
                    speedup = base_time / results[natoms]['mean']
                    print(f"{speedup:>18.2f}x", end="")
                else:
                    print(f"{'N/A':>18}", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACE potential interfaces")
    parser.add_argument("--model", type=str, help="Path to ACE model JSON file")
    parser.add_argument("--library", type=str, required=True, help="Path to compiled library (.so)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5],
                       help="Supercell sizes to test (default: 2 3 4 5)")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of timed iterations (default: 5)")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8],
                       help="Thread counts to test (default: 1 4 8)")
    parser.add_argument("--skip-socket", action="store_true",
                       help="Skip socket calculator benchmarks")
    parser.add_argument("--skip-julia", action="store_true",
                       help="Skip native Julia benchmarks")
    parser.add_argument("--lammps", type=str, default="lmp",
                       help="Path to LAMMPS executable (default: lmp)")
    parser.add_argument("--plugin", type=str,
                       help="Path to ACE LAMMPS plugin (.so)")
    parser.add_argument("--skip-lammps", action="store_true",
                       help="Skip LAMMPS benchmarks")
    args = parser.parse_args()

    print("ACE Potential Interface Benchmark")
    print("==================================")
    print(f"Supercell sizes: {args.sizes}")
    print(f"Iterations per size: {args.iterations}")
    print(f"Thread counts: {args.threads}")

    # Detect species from library
    from ase_ace import ACELibraryCalculator
    from ase.data import chemical_symbols
    calc = ACELibraryCalculator(args.library)
    species_z = calc.species[0]  # Use first species
    species = chemical_symbols[species_z]
    print(f"Library species: {species} (Z={species_z})")

    # Create test structures
    print("\nCreating test structures...")
    structures = create_structures(args.sizes, species=species)
    sizes = sorted(structures.keys())
    print(f"Structure sizes: {sizes} atoms")

    all_results = {}

    # Benchmark library calculator (single-threaded)
    print("\n" + "-" * 60)
    print("Benchmarking ACELibraryCalculator (single-threaded)...")
    lib_results = benchmark_library_calculator(args.library, structures, args.iterations)
    if lib_results:
        all_results["Library (1T)"] = lib_results
        print_results(lib_results, "ACELibraryCalculator (1 thread)")
    else:
        print("Library calculator not available")

    # Benchmark LAMMPS with different thread counts
    if not args.skip_lammps and args.plugin:
        for threads in args.threads:
            print("\n" + "-" * 60)
            print(f"Benchmarking LAMMPS ({threads} OpenMP threads)...")
            lammps_results = benchmark_lammps(
                args.library, args.plugin, args.lammps, structures,
                args.iterations, omp_threads=threads
            )
            if lammps_results:
                all_results[f"LAMMPS ({threads}T)"] = lammps_results
                print_results(lammps_results, f"LAMMPS ({threads} OpenMP threads)")
            else:
                print(f"LAMMPS benchmark failed with {threads} threads")

    # Benchmark socket calculator with different thread counts
    if not args.skip_socket and args.model:
        for threads in args.threads:
            print("\n" + "-" * 60)
            print(f"Benchmarking ACECalculator (socket, {threads} threads)...")
            os.environ['JULIA_NUM_THREADS'] = str(threads)
            socket_results = benchmark_socket_calculator(
                args.model, structures, threads, args.iterations
            )
            if socket_results:
                all_results[f"Socket ({threads}T)"] = socket_results
                print_results(socket_results, f"ACECalculator (socket, {threads} threads)")
            else:
                print(f"Socket calculator failed with {threads} threads")

    # Benchmark native Julia with different thread counts
    if not args.skip_julia and args.model:
        for threads in args.threads:
            print("\n" + "-" * 60)
            print(f"Benchmarking Native Julia ({threads} threads)...")
            julia_results = benchmark_native_julia(
                args.model, structures, threads, args.iterations
            )
            if julia_results:
                all_results[f"Julia ({threads}T)"] = julia_results
                print_results(julia_results, f"Native Julia ({threads} threads)")
            else:
                print(f"Native Julia benchmark failed with {threads} threads")

    # Print comparison
    if len(all_results) > 1:
        print_comparison(all_results, sizes)


if __name__ == "__main__":
    main()
