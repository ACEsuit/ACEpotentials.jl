#=
Python Calculator Tests

Runs Python tests via Julia's run() command.
Tests the Python/ASE calculator wrapper for the compiled ACE library.
=#

using Test

@testset "Python Calculator" verbose=true begin
    build_dir = joinpath(TEST_DIR, "build")
    lib_path = joinpath(build_dir, "libace_test.so")

    if !isfile(lib_path)
        @test_skip "Library not compiled - skipping Python tests"
        return
    end

    python_test_dir = joinpath(TEST_DIR, "python")
    python_calc_dir = joinpath(EXPORT_DIR, "python")

    # Set environment for Python tests
    env = copy(ENV)
    env["ACE_LIB_PATH"] = lib_path
    env["PYTHONPATH"] = python_calc_dir * ":" * get(ENV, "PYTHONPATH", "")

    # Find Julia runtime libraries for LD_LIBRARY_PATH
    julia_lib_dir = joinpath(Sys.BINDIR, "..", "lib")
    if haskey(env, "LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = julia_lib_dir * ":" * env["LD_LIBRARY_PATH"]
    else
        env["LD_LIBRARY_PATH"] = julia_lib_dir
    end

    @testset "Library Loading" begin
        # Test that Python can load the library
        result = try
            read(setenv(`python3 -c "
import os
import ctypes
lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)
print('ok')
"`, env), String)
        catch e
            "error: $e"
        end
        @test strip(result) == "ok"
    end

    @testset "Utility Functions" begin
        # Test cutoff and species info
        result = read(setenv(`python3 -c "
import os
import ctypes
lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)
lib.ace_get_cutoff.restype = ctypes.c_double
lib.ace_get_n_species.restype = ctypes.c_int
lib.ace_get_species.restype = ctypes.c_int
lib.ace_get_species.argtypes = [ctypes.c_int]

cutoff = lib.ace_get_cutoff()
nspecies = lib.ace_get_n_species()
species = lib.ace_get_species(1)

print(f'{cutoff:.2f},{nspecies},{species}')
"`, env), String)

        parts = split(strip(result), ",")
        cutoff = parse(Float64, parts[1])
        nspecies = parse(Int, parts[2])
        species = parse(Int, parts[3])

        @test cutoff > 0 && cutoff < 10
        @test nspecies == 1
        @test species == 14
    end

    @testset "Energy Calculation" begin
        # Test energy calculation with Python ctypes
        result = read(setenv(`python3 -c "
import os
import ctypes
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)

# Set up function signatures
lib.ace_energy.restype = ctypes.c_double
lib.ace_energy.argtypes = [
    ctypes.c_int,  # natoms
    ctypes.POINTER(ctypes.c_int),  # species
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.POINTER(ctypes.c_double),  # cell
    ctypes.POINTER(ctypes.c_int),  # pbc
]

# Si diamond structure (8 atoms)
a = 5.43
positions = np.array([
    [0.00, 0.00, 0.00], [0.50, 0.50, 0.00], [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
    [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
], dtype=np.float64) * a

cell = np.eye(3, dtype=np.float64) * a
species = np.array([14]*8, dtype=np.int32)
pbc = np.array([1, 1, 1], dtype=np.int32)

natoms = 8
E = lib.ace_energy(
    natoms,
    species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
)
print(f'{E:.10f}')
"`, env), String)

        E = parse(Float64, strip(result))
        @test isfinite(E)
        # Test model may not have negative cohesive energy (depends on reference)
        @test abs(E/8) < 100  # Reasonable per-atom energy

        # Store for comparison with LAMMPS
        TEST_ARTIFACTS["python_energy_8atom"] = E
    end

    @testset "Forces Calculation" begin
        # Test forces with finite difference verification
        result = read(setenv(`python3 -c "
import os
import ctypes
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)

lib.ace_energy.restype = ctypes.c_double
lib.ace_energy.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int)
]

lib.ace_energy_forces.restype = ctypes.c_double
lib.ace_energy_forces.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
]

a = 5.43
positions = np.array([
    [0.00, 0.00, 0.00], [0.50, 0.50, 0.00], [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
    [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
], dtype=np.float64) * a

# Add small random perturbation for non-zero forces
np.random.seed(42)
positions += np.random.randn(*positions.shape) * 0.01

cell = np.eye(3, dtype=np.float64) * a
species = np.array([14]*8, dtype=np.int32)
pbc = np.array([1, 1, 1], dtype=np.int32)
natoms = 8

forces = np.zeros((natoms, 3), dtype=np.float64)

E = lib.ace_energy_forces(
    natoms,
    species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)

# Finite difference check on first atom
h = 1e-6
fd_forces = np.zeros(3)
for alpha in range(3):
    pos_p = positions.copy()
    pos_m = positions.copy()
    pos_p[0, alpha] += h
    pos_m[0, alpha] -= h

    E_p = lib.ace_energy(
        natoms, species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pos_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    E_m = lib.ace_energy(
        natoms, species.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pos_m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pbc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    fd_forces[alpha] = -(E_p - E_m) / (2*h)

max_err = np.max(np.abs(forces[0] - fd_forces))
print(f'{max_err:.2e}')
"`, env), String)

        max_err = parse(Float64, strip(result))
        # FD with h=1e-6 typically gives ~1e-4 to 1e-3 accuracy
        @test max_err < 1e-2
    end

    @testset "ASE Calculator" begin
        # Test full ASE calculator interface
        result = read(setenv(`python3 -c "
import os
import sys
sys.path.insert(0, os.environ.get('PYTHONPATH', '').split(':')[0])

from ace_calculator import ACECalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create Si diamond structure
atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = calc

# Get energy, forces, stress
E = atoms.get_potential_energy()
F = atoms.get_forces()
S = atoms.get_stress()

# Basic sanity checks
assert np.isfinite(E), 'Energy not finite'
assert np.all(np.isfinite(F)), 'Forces not finite'
assert np.all(np.isfinite(S)), 'Stress not finite'

# Perfect crystal should have near-zero forces
max_force = np.abs(F).max()
assert max_force < 1e-6, f'Max force too large: {max_force}'

print('ok')
"`, env), String)

        @test strip(result) == "ok"
    end

    @testset "ASE MD Energy Conservation" begin
        # Run short NVE MD and check energy conservation
        result = read(setenv(`python3 -c "
import os
import sys
sys.path.insert(0, os.environ.get('PYTHONPATH', '').split(':')[0])

from ace_calculator import ACECalculator
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create 2x2x2 supercell for MD
atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
atoms.calc = calc

# Initialize velocities at 100 K (low temp for stability)
MaxwellBoltzmannDistribution(atoms, temperature_K=100)
momenta = atoms.get_momenta()
momenta -= momenta.mean(axis=0)
atoms.set_momenta(momenta)

# Run NVE
dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

energies = []
for i in range(20):
    dyn.run(5)
    e_kin = atoms.get_kinetic_energy()
    e_pot = atoms.get_potential_energy()
    energies.append(e_kin + e_pot)

energies = np.array(energies)
drift = np.abs(energies[-1] - energies[0])
std = np.std(energies)

# Energy should be conserved to ~1e-4 eV for this short run
print(f'{drift:.6e},{std:.6e}')
"`, env), String)

        parts = split(strip(result), ",")
        drift = parse(Float64, parts[1])
        std = parse(Float64, parts[2])

        @test drift < 0.1  # Allow some drift over 100 steps
        @test std < 0.1    # Reasonable energy fluctuation
    end
end
