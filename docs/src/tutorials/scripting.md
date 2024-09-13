
# Basic Shell Workflow

This short introduces a basic workflow where models are specified via 
JSON files and model fitting is achieved via a shell script. 

At the start of a new project we 
- create a project folder 
- activate a Julia project in that folder 
- add `ACEpotentials` to the Julia project 
- generate a fitting script and example model specification file

```shell 
mkdir myace
cd myace
julia --project=. -e 'using Pkg; Pkg.add("ACEpotentials"); using ACEpotentials; ACEpotentials.copy_runfit(@__DIR__())'
```

This should create two new files in the `myace` folder: 
- `runfit.jl`
- `example_params.json`
Copy (or move) the `example_params.json` file to a new filename, e.g. 

```shell 
cp example_params.json myace_params_1.json
```

then edit that file to specify the model hyperparameters, 
the fitting method (see also [`ACEfit.jl`](https://github.com/ACEsuit/ACEfit.jl)), and the path to the dataset (or, datasets if validation 
and or test sets are also provided). To produce a fit, use 

```shell 
julia --project=. runfit.jl -p myace_params_1.json -o results_1
```

This will write all outputs to the `./results_1` folder, in particular `results.json` which contains the model specification, the fitted model parameters, and a dictionary of computed errors (rmse, mae). The list of required outputs and the output filename(s) can be changed in the model spec json. 

