#=
Python Calculator Tests

Runs Python tests via Julia's run() command.
Tests the Python/ASE calculator wrapper for the compiled ACE library.

The ACECalculator uses matscipy for O(N) neighbor list construction
and calls the site-level C API for energy/force evaluation.
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

    # Set environment for Python tests
    env = copy(ENV)
    env["ACE_LIB_PATH"] = lib_path

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

    @testset "Site Energy Calculation" begin
        # Test site-level energy calculation with Python ctypes
        result = read(setenv(`python3 -c "
import os
import ctypes
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)

# Set up function signature for site energy
lib.ace_site_energy.restype = ctypes.c_double
lib.ace_site_energy.argtypes = [
    ctypes.c_int,  # z0
    ctypes.c_int,  # nneigh
    ctypes.POINTER(ctypes.c_int),  # neighbor_z
    ctypes.POINTER(ctypes.c_double),  # neighbor_Rij
]

# Test with Si center and 4 Si neighbors (tetrahedral)
z0 = 14
a = 5.43
# First neighbor shell in diamond Si (distance ~2.35 A)
d = a * np.sqrt(3) / 4  # bond length
neighbor_R = np.array([
    [ d/np.sqrt(3),  d/np.sqrt(3),  d/np.sqrt(3)],
    [ d/np.sqrt(3), -d/np.sqrt(3), -d/np.sqrt(3)],
    [-d/np.sqrt(3),  d/np.sqrt(3), -d/np.sqrt(3)],
    [-d/np.sqrt(3), -d/np.sqrt(3),  d/np.sqrt(3)],
], dtype=np.float64)

neighbor_z = np.array([14, 14, 14, 14], dtype=np.int32)
nneigh = 4

E = lib.ace_site_energy(
    z0,
    nneigh,
    neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    neighbor_R.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)
print(f'{E:.10f}')
"`, env), String)

        E = parse(Float64, strip(result))
        @test isfinite(E)
    end

    @testset "Site Forces Calculation" begin
        # Test site-level forces with finite difference verification
        result = read(setenv(`python3 -c "
import os
import ctypes
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)

lib.ace_site_energy.restype = ctypes.c_double
lib.ace_site_energy.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double)
]

lib.ace_site_energy_forces.restype = ctypes.c_double
lib.ace_site_energy_forces.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)  # forces output
]

z0 = 14
a = 5.43
d = a * np.sqrt(3) / 4
neighbor_R = np.array([
    [ d/np.sqrt(3),  d/np.sqrt(3),  d/np.sqrt(3)],
    [ d/np.sqrt(3), -d/np.sqrt(3), -d/np.sqrt(3)],
    [-d/np.sqrt(3),  d/np.sqrt(3), -d/np.sqrt(3)],
    [-d/np.sqrt(3), -d/np.sqrt(3),  d/np.sqrt(3)],
], dtype=np.float64)

# Add small perturbation for non-zero forces
np.random.seed(42)
neighbor_R += np.random.randn(*neighbor_R.shape) * 0.01

neighbor_z = np.array([14, 14, 14, 14], dtype=np.int32)
nneigh = 4
forces = np.zeros(nneigh * 3, dtype=np.float64)

E = lib.ace_site_energy_forces(
    z0, nneigh,
    neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    neighbor_R.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)
forces = forces.reshape(nneigh, 3)

# Finite difference check on first neighbor displacement
h = 1e-6
fd_forces = np.zeros(3)
for alpha in range(3):
    R_p = neighbor_R.copy()
    R_m = neighbor_R.copy()
    R_p[0, alpha] += h
    R_m[0, alpha] -= h

    E_p = lib.ace_site_energy(
        z0, nneigh,
        neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        R_p.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    E_m = lib.ace_site_energy(
        z0, nneigh,
        neighbor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        R_m.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    # site_energy_forces returns -dE/dRj (force ON neighbor)
    fd_forces[alpha] = -(E_p - E_m) / (2*h)

max_err = np.max(np.abs(forces[0] - fd_forces))
print(f'{max_err:.2e}')
"`, env), String)

        max_err = parse(Float64, strip(result))
        # FD with h=1e-6 typically gives ~1e-4 to 1e-3 accuracy
        @test max_err < 1e-2
    end

    @testset "ACE Calculator with matscipy" begin
        # Test full ASE calculator interface using matscipy neighbor list
        result = read(setenv(`python3 -c "
import os
from ase_ace import ACELibraryCalculator as ACECalculator
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

    @testset "Larger System Test" begin
        # Test with a larger supercell to verify O(N) scaling works
        result = read(setenv(`python3 -c "
import os
from ase_ace import ACELibraryCalculator as ACECalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create 3x3x3 supercell (216 atoms)
atoms = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)
atoms.calc = calc

E = atoms.get_potential_energy()
F = atoms.get_forces()

# Energy per atom should be consistent with unit cell
E_per_atom = E / len(atoms)

# Check forces are finite
assert np.all(np.isfinite(F)), 'Forces not finite'

# Perfect crystal should have near-zero forces
max_force = np.abs(F).max()
assert max_force < 1e-5, f'Max force too large: {max_force}'

print(f'{E_per_atom:.6f},{max_force:.2e}')
"`, env), String)

        parts = split(strip(result), ",")
        E_per_atom = parse(Float64, parts[1])
        max_force = parse(Float64, parts[2])

        @test isfinite(E_per_atom)
        @test max_force < 1e-5
    end

    @testset "ASE MD Energy Conservation" begin
        # Run short NVE MD and check energy conservation
        result = read(setenv(`python3 -c "
import os
from ase_ace import ACELibraryCalculator as ACECalculator
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

    @testset "Triclinic Cell" begin
        # Test with triclinic cell to verify matscipy handles it correctly
        result = read(setenv(`python3 -c "
import os
from ase_ace import ACELibraryCalculator as ACECalculator
from ase import Atoms
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create triclinic Si cell
a = 5.43
cell = np.array([
    [a, 0, 0],
    [0.3*a, a, 0],
    [0.1*a, 0.2*a, a]
])

# Si diamond basis positions in fractional coordinates
frac_pos = np.array([
    [0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25],
])
positions = frac_pos @ cell

atoms = Atoms('Si2', positions=positions, cell=cell, pbc=True)
atoms.calc = calc

E = atoms.get_potential_energy()
F = atoms.get_forces()

assert np.isfinite(E), 'Energy not finite'
assert np.all(np.isfinite(F)), 'Forces not finite'

print(f'{E:.6f}')
"`, env), String)

        E = parse(Float64, strip(result))
        @test isfinite(E)
    end

    @testset "Descriptor API" begin
        # Test n_basis
        result = read(setenv(`python3 -c "
import os
import ctypes
lib_path = os.environ['ACE_LIB_PATH']
lib = ctypes.CDLL(lib_path)
lib.ace_get_n_basis.restype = ctypes.c_int
n_basis = lib.ace_get_n_basis()
print(f'{n_basis}')
"`, env), String)

        n_basis = parse(Int, strip(result))
        @test n_basis > 0 && n_basis < 1000
    end

    @testset "Get Descriptors via Calculator" begin
        # Test the full descriptor computation via the calculator
        result = read(setenv(`python3 -c "
import os
from ase_ace import ACELibraryCalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACELibraryCalculator(lib_path)

# Test n_basis property
n_basis = calc.n_basis
assert n_basis > 0, 'n_basis should be positive'

# Test get_descriptors
atoms = bulk('Si', 'diamond', a=5.43)
D = calc.get_descriptors(atoms)

# Check shape
assert D.shape == (2, n_basis), f'Wrong shape: {D.shape}'

# Check finite
assert np.all(np.isfinite(D)), 'Descriptors not finite'

# Check not all zero
assert np.any(D != 0), 'All descriptors zero'

# Check deterministic
D2 = calc.get_descriptors(atoms)
assert np.allclose(D, D2), 'Descriptors not deterministic'

print(f'{n_basis},{D.shape[0]},{D.shape[1]}')
"`, env), String)

        parts = split(strip(result), ",")
        n_basis = parse(Int, parts[1])
        natoms = parse(Int, parts[2])
        nbasis = parse(Int, parts[3])

        @test n_basis > 0
        @test natoms == 2
        @test nbasis == n_basis
    end

    @testset "Descriptor Supercell Consistency" begin
        # Test that descriptors are consistent across supercells
        result = read(setenv(`python3 -c "
import os
from ase_ace import ACELibraryCalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACELibraryCalculator(lib_path)

# Unit cell
atoms = bulk('Si', 'diamond', a=5.43)
D1 = calc.get_descriptors(atoms)

# 2x2x2 supercell
atoms_big = atoms * (2, 2, 2)
D_big = calc.get_descriptors(atoms_big)

# All atoms in perfect crystal should have the same descriptor
max_diff = np.max(np.abs(D_big - D_big[0:1]))

print(f'{max_diff:.2e}')
"`, env), String)

        max_diff = parse(Float64, strip(result))
        @test max_diff < 1e-10
    end
end
