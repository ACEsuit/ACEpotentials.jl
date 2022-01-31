
using ACE1pack

# for now just make a dictionary and make a fit!
fit_params = Dict(
    "data_params" => Dict(
        "xyz_filename" => "/Users/elena/.julia/dev/ACE1pack/test/files/TiAl_tutorial_DB_tenth.xyz",
        "energy_key" => "energy",
        "force_key" => "force",
        "virial_key" => "virial"),
    "rpi_basis_params" => Dict(
        "species" => ["Ti", "Al"],
        "N" => 3, 
        "maxdeg" => 6,
        "r0" => 2.88,  # `0.5(rnn(:Ti)+ rnn(:Al))`
        "radbasis" => Dict(
            "rcut" => 5.0,
            "rin" => 1.44, # 0.5 *r0
            "pcut" => 2, 
            "pin" => 2),
        "transform" => Dict(
           "type" => "polynomial",
           "p" => 2, 
           "r0" => 2.88),
        "degree" => Dict(
            "type" => "sparse",
            "wL" => 1.5,
            "csp" => 1.0,
            "chc" => 0.0,
            "ahc" => 0.0,
            "bhc" => 0.0,
            "p" => 1.0)),
    "pair_basis_params" => Dict(
        "species" => ["Ti", "Al"],
        "maxdeg" => 6, 
        "r0" => 2.88,
        "radbasis" => Dict(
            "rcut" => 5.0,
            "rin" => 1.44, # 0.5 *r0
            "pcut" => 2, 
            "pin" => 2), 
        "transform" => Dict(
            "type" => "polynomial",
            "p" => 2, 
            "r0" => 2.88)),
    "solver_params" => Dict(
        "solver" => "lsqr",
        "lsqr_damp" => 5e-3,
        "lsqr_atol" => 1e-6,
        "rlap_scal" => 3.0),
    "E0_dict" => Dict(          # from tial tutorial
        "Ti" => -1586.0195, 
        "Al" => -105.5954),
    "weights" => Dict(
        "default" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ),
        "FLD_TiAl" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 )),
    "ACE_filename" => "tial_ace.json",
    "db_filename" => "tial_db",    # confusingly just prefix, not full filename; TODO
    "error_table" => true
)

@show fit_params
ACE1pack.fit(fit_params)