# Saving and Loading Potentials

!!! warning 
    Saving and loading potentials is currently only supported for the 
    a workflow that uses JSON / dictionaries to specify models. For anything 
    "too creative" the user is responsible for ensuring reproducability. 

### General Principles 

Loading a saved potentials is only guaranteed if the Julia environment 
remains the same. A new project should therefore always work with a specified 
`Project.toml` and `Manifest.toml`. See [out Pkg intro](pkg.md) for a brief 
introduction and references to further details. 

If the manifest changes, but the ACEpotentials version remains the same or 
a backward compatible update (cf semver) then in principle a saved potential 
should remain loadable. We cannot guarntee this but would consider it a bug 
if this is not the case.

Normally, we save the entire Julia environment together with a fitted 
potential. This way it should always be possible to reconstruct the 
environment and hence the potential. More details follow. 


### Saving JSON-specified potentials

If using the `runfit.jl` script, then an output folder is specified, where 
all information including the full model specification and model parameters 
are stored as a JSON file (together with other meta-information). 

### Loading a JSON-specified potential

Suppose the result of `runfit.jl` (or an analogous approach) is saved to 
`path/result.json`. If the original or a compatible Julia environment is 
activated, then 
```julia
model = ACEpotentials.load_model("path/result.json")
```
will return a `model::ACEPotential` structure that should be equivalent 
to the original fitted potential. 

### Recovering the Julia environment 

At the moment, this process is not implemented, but the `result.json` file
can loaded into a dictionary which can then be investigated to manually 
reconstruct the environment and then load the potential as described in 
the previous section. 

