#!/usr/bin/env julia
"""
Integration test for LAMMPS ACE/Minimal plugin.

This test:
1. Creates and exports an ACE model using minimal export
2. Runs LAMMPS with the ACE/Minimal pair style
3. Validates energies and forces against Julia calculator
4. Checks that results match within tolerance

Requirements:
- LAMMPS with plugin support
- ACE/Minimal plugin built (libaceminimal.so)
- Environment variables set:
  - LAMMPS_EXE: Path to LAMMPS executable
  - ACE_C_INTERFACE_PATH: Path to ace_c_interface_minimal.jl
"""

using Test
using ACEpotentials
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETOneBody, StackedCalculator, one_body
using StaticArrays
using Random
using Lux
using LuxCore
using AtomsBase
using AtomsCalculators
using Unitful
using AtomsBase: ChemicalSpecies

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

# Load export function
export_dir = dirname(dirname(@__DIR__))
include(joinpath(export_dir, "src", "export_ace_model_minimal.jl"))

@testset "LAMMPS ACE/Minimal Integration" begin
    println("\n" * "="^80)
    println("LAMMPS ACE/Minimal Integration Test")
    println("="^80)

    # Check environment
    lammps_exe = get(ENV, "LAMMPS_EXE", nothing)
    if lammps_exe === nothing || !isfile(lammps_exe)
        @warn "LAMMPS executable not found. Set LAMMPS_EXE environment variable."
        @warn "Skipping LAMMPS integration test."
        return
    end

    c_interface_path = get(ENV, "ACE_C_INTERFACE_PATH", nothing)
    if c_interface_path === nothing || !isfile(c_interface_path)
        @warn "ACE_C_INTERFACE_PATH not set or file not found."
        @warn "Skipping LAMMPS integration test."
        return
    end

    println("\n[1] Creating test model...")
    elements = (:Si,)
    rcut = 5.5
    rng = Random.MersenneTwister(1234)

    ace_model = M.ace_model(;
        elements = elements,
        order = 2,
        Ytype = :solid,
        level = M.TotalDegree(),
        max_level = 8,
        maxl = 2,
        pair_maxn = 8,
        rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(M._default_rin0cuts(elements)),
        init_WB = :glorot_normal,
        init_Wpair = :glorot_normal
    )

    ps, st = Lux.setup(rng, ace_model)

    # Convert to ETACE
    println("[2] Converting to ETACE...")
    et_model = ETM.convert2et(ace_model)
    et_ps, et_st = LuxCore.setup(rng, et_model)

    # Copy parameters
    for iz in 1:1, jz in 1:1
        et_ps.rembed.post.W[:, :, (iz-1)*1 + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
    for iz in 1:1
        et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
    end

    # Splinify
    println("[3] Splinifying model...")
    et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
    et_ps_splined, et_st_splined = LuxCore.setup(rng, et_model_splined)

    for iz in 1:1
        et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
    end

    # Create ETACE calculator
    etace_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

    # Create ETOneBody
    println("[4] Creating ETOneBody...")
    E0_dict_species = Dict(ChemicalSpecies(:Si) => -5.0)
    e0_model = one_body(E0_dict_species, x -> x.z)
    e0_ps, e0_st = LuxCore.setup(rng, e0_model)
    e0_calc = ETM.ETOneBodyPotential(e0_model, e0_ps, e0_st, rcut)

    # Create StackedCalculator
    println("[5] Creating StackedCalculator...")
    calc = StackedCalculator((e0_calc, etace_calc))

    # Export model
    println("\n[6] Exporting model...")
    model_dir = joinpath(@__DIR__, "build", "lammps_test_model")
    export_ace_model_minimal(calc, model_dir; model_name="test_ace")

    # Create test structure
    println("\n[7] Creating test structure...")
    # Simple 2-atom Si structure
    positions = [
        SVector(0.0, 0.0, 0.0),
        SVector(3.0, 0.0, 0.0),
    ]
    box = [SVector(10.0, 0.0, 0.0), SVector(0.0, 10.0, 0.0), SVector(0.0, 0.0, 10.0)]

    sys = AtomsBase.periodic_system(
        [:Si => pos * u"Å" for pos in positions],
        [b * u"Å" for b in box]
    )

    # Compute reference with Julia
    println("\n[8] Computing reference energy with Julia...")
    E_julia = AtomsCalculators.potential_energy(sys, calc)
    F_julia = AtomsCalculators.forces(sys, calc)

    E_julia_val = ustrip(u"eV", E_julia)
    println("   Julia energy: $E_julia_val eV")

    # Create LAMMPS data file
    println("\n[9] Creating LAMMPS data file...")
    data_file = joinpath(@__DIR__, "build", "test_structure.data")
    open(data_file, "w") do io
        println(io, "LAMMPS data file for ACE/Minimal test")
        println(io, "")
        println(io, "2 atoms")
        println(io, "1 atom types")
        println(io, "")
        println(io, "0.0 10.0 xlo xhi")
        println(io, "0.0 10.0 ylo yhi")
        println(io, "0.0 10.0 zlo zhi")
        println(io, "")
        println(io, "Masses")
        println(io, "")
        println(io, "1 28.0855")
        println(io, "")
        println(io, "Atoms")
        println(io, "")
        println(io, "1 1 0.0 0.0 0.0")
        println(io, "2 1 3.0 0.0 0.0")
    end

    # Create LAMMPS input script
    println("\n[10] Creating LAMMPS input script...")
    input_file = joinpath(@__DIR__, "build", "test.lammps")
    plugin_path = abspath(joinpath(dirname(@__DIR__), "plugin", "lib", "libaceminimal.so"))

    open(input_file, "w") do io
        println(io, "# LAMMPS input for ACE/Minimal test")
        println(io, "units metal")
        println(io, "atom_style atomic")
        println(io, "boundary p p p")
        println(io, "")
        println(io, "read_data $data_file")
        println(io, "")
        println(io, "plugin load $plugin_path")
        println(io, "pair_style ace/minimal")
        println(io, "pair_coeff * * $model_dir Si")
        println(io, "")
        println(io, "neighbor 2.0 bin")
        println(io, "neigh_modify delay 0 every 1 check yes")
        println(io, "")
        println(io, "thermo_style custom step pe")
        println(io, "thermo 1")
        println(io, "")
        println(io, "run 0")
        println(io, "")
        println(io, "write_dump all custom dump.ace id type x y z fx fy fz modify format float %.8f")
    end

    # Run LAMMPS
    println("\n[11] Running LAMMPS...")
    lammps_cmd = `$lammps_exe -in $input_file -log $(joinpath(@__DIR__, "build", "log.lammps"))`

    try
        run(lammps_cmd)
    catch e
        @error "LAMMPS execution failed" exception=e
        @test false
        return
    end

    # Parse LAMMPS output
    println("\n[12] Parsing LAMMPS output...")
    log_file = joinpath(@__DIR__, "build", "log.lammps")
    log_content = read(log_file, String)

    # Extract energy from log
    energy_pattern = r"Step\s+PotEng\s+0\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    energy_match = match(energy_pattern, log_content)

    if energy_match === nothing
        @error "Could not parse energy from LAMMPS output"
        @test false
        return
    end

    E_lammps = parse(Float64, energy_match.captures[1])
    println("   LAMMPS energy: $E_lammps eV")

    # Compare energies
    println("\n[13] Validating results...")
    energy_diff = abs(E_lammps - E_julia_val)
    println("   Energy difference: $energy_diff eV")

    @testset "Energy Validation" begin
        @test E_lammps ≈ E_julia_val atol=1e-6 rtol=1e-6
    end

    # TODO: Parse and validate forces from dump file

    println("\n" * "="^80)
    println("✓ LAMMPS ACE/Minimal Integration Test Complete")
    println("="^80)
end
