# OLD (!) Command line interface

The (most likely) easiest way to fit an ACE potential is via the command line from a JSON or YAML parameters file: 

```bash
julia .../outdated/scripts/ace_fit.jl --params ace_params.json
```

In addition to parameters' file, `ace_fit.jl` takes an optional `--dry-run` flag. If it is given, a `.size` file is produced with the shape of the design matrix, useful for estimating the time and memory requirements before submitting an actual fit. Finally, there is a `--num-blas-threads` option for setting the number of BLAS threads to use (for fitting). 

For the script to use the correct Julia environment, `JULIA_PROJECT` (a path) must be set to the folder where Julia's `Project.toml` and `Manifest.toml` are. 

Below are examples of the parameters' files. The first one gives only the mandatory values and the second one has all of the default values filled in. For details on specific values see the appropriate pages of [ACEpotentials Internals](../ACEpotentials/acepotentials_overview.md). Explanation of the top-level dictionary, with links to the nested dictionaries therein are in [Fitting ACE](../outdated/fit.md). 


Mandatory parameters

```

{
    "e0": {
        "Ti": -1586.0195,
        "Al": -105.5954},
    "data": {
        "fname": "training_set.xyz"},
    "solver": {"type": "rrqr"},
    "basis": {
        "main_ace": {
            "type": "ace",
            "species": ["Ti", "Al"]},
            "N": 2,
            "maxdeg": 10,
        "main_pair": {
            "species": ["Ti","Al"]
            "maxdeg": 4,
        }
    }
}

```

Parameters with all default values

```

{
    "e0": {
        "Ti": -1586.0195,
        "Al": -105.5954},
    "weights": {
        "default": {
            "E": 1.0,
            "F": 1.0}},
            "V": 1.0,
    "P": null,
    "fit_from_LSQ_DB": false,
    "data": {
        "fname": "training_set.xyz",
        "energy_key": "dft_energy",
        "force_key": "dft_force",
        "virial_key": "dft_virial"},
    "solver": {
        "type": "rrqr"},
        "tol": 1.0e-5,
    "basis": {
        "main_ace": {
            "type": "ace",
            "species": ["Ti", "Al"]},
            "N": 2,
            "maxdeg": 10,
            "r0": 2.5,
            "radial": {
                "rcut": 5.0,
                "rin": 0.5,
                "r0": 2.5,
                "pin": 2,
                "pcut": 2,
                "type": "radial"},
            "degree": "degree",
            "transform": {
                "r0": 2.5,
                "type": "polynomial",
                "p": 2},
        "main_pair": {
            "type": "pair",
            "species": ["Ti", "Al"]}},
            "rcut": 5.0,
            "rin": 0.0,
            "maxdeg": 4,
            "r0": 2.5,
            "pin": 0,
            "pcut": 2,
            "transform": {
                "r0": 2.5,
                "type": "polynomial",
                "p": 2},
    "LSQ_DB_fname_stem": "",
    "ACE_fname": "ACE_fit.json"
}

```
