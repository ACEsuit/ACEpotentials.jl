
@testset "Fit ACE" begin

    using ACE1pack, JuLIP

    @info("test full fit")

    fit_filename = @__DIR__() * "/files/TiAl_tutorial_DB_tenth.xyz"
    species = [:Ti, :Al]
    r0 = sum(rnn(sp) for sp in species) / length(species)

    data = data_params(fname = fit_filename,
        energy_key = "energy",
        force_key = "force",
        virial_key = "virial")

    rpi_radbasis = radbasis_params(rcut = 5.5, rin = 0.6 * r0, pin = 2)
    rpi_basis = rpi_basis_params(species = species, N = 3, maxdeg = 6, r0 = r0, radbasis = rpi_radbasis)

    # pair_radbasis = radbasis_params(rcut = 7.0, pcut = 1, pin = 0, rin = 0.0)
    pair_radbasis = radbasis_params(rcut = 7.0, pcut = 1, pin = 0, rin = 0.5 * r0)

    pair_basis = pair_basis_params(species = species, maxdeg = 6, r0 = r0, radbasis = pair_radbasis)

    solver = solver_params(solver = :lsqr, rlap_scal = 3.0)

    # symbols for species (e.g. :Ti) would work as well
    e0 = Dict("Ti" => -1586.0195, "Al" => -105.5954)

    weights = Dict(
        "default" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0),
        "FLD_TiAl" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0),
        "TiAl_T5000" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0))


    fit_params = ace_params(
        data = data,
        rpi_basis = rpi_basis,
        pair_basis = pair_basis,
        solver = solver,
        e0 = e0,
        weights = weights)

    IP, lsqinfo = ACE1pack.fit_ace(fit_params)

    expected_errors = load_dict(@__DIR__() * "/files/expected_fit_errors.json")
    errors = lsqinfo["errors"]

    for error_type in keys(errors)
        for config_type in keys(errors[error_type])
            for property in keys(errors[error_type][config_type])
                @test errors[error_type][config_type][property] ≈  
                expected_errors[error_type][config_type][property]
            end
        end
    end

end
