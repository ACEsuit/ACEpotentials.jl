# Command line interface

Perhaps the easiest way to fit an ACE potential is via the command line from a JSON or YAML parameters file: 

```bash
julia .../ACEpotentials.jl/scripts/ace_fit.jl --params ace_params.json
```

In addition to parameters' file, `ace_fit.jl` takes an optional `--dry-run` flag. If it is given, a `.size` file is produced with the shape of the design matrix, useful for estimating the time and memory requirements before submitting an actual fit. Finally, there is a `--num-blas-threads` option for setting the number of BLAS threads to use (for fitting). 

For the script to use the correct Julia environment, `JULIA_PROJECT` (a path) must be set to the folder where Julia's `Project.toml` and `Manifest.toml` are. 

Below are examples of the parameters' files. The first one gives only the mandatory values and the second one has all of the default values filled in. For details on specific values see the appropriate pages of [ACEpotentials Internals](./ACEpotentials/acepotentials_overview.md). Explanation of the top-level dictionary, with links to the nested dictionaries therein are in [Fitting ACE](./ACEpotentials/fit.md). 


Example parameters

(One can access the test data at the location printed by `julia --project=@. -e "using ACEpotentials; using LazyArtifacts; println(joinpath(artifact\"Si_tiny_dataset\", \"Si_tiny.xyz\"))"`.)

```
{
    "elements": ["Si"],
    "order": 3,
    "totaldegree": 12,
    "rcut": 5.0,
    "datafile": "Si_tiny.xyz",
    "solver": "BayesianLinearRegression",
    "energy_key": "dft_energy",
    "force_key": "dft_force",
    "virial_key": "dft_virial"
}
```

### TODO

All parameters with default values

```

{
}

```
