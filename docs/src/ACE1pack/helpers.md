# Helper Functions

Probably the most useful utility function is `fill_defaults()` which takes in a dictionary compatible with one of the `*params()` functions and recursively fills in the default values for non-mandatory functions. 

The (incomplete) dictionaries may read from a JSON or YAML file. Creating these files with another programming language and then calling a short Julia script to read in and fit using these parameters is the expected way of interfacing with ACE1 from other languages. 

```julia
params = load_dict("params.json")
# or 
params = load_dict("params.yaml")
```


Some of the ACE basis parameters dictionaries keys and values may be 2-tuples (specifically, the "multitransform" and "sparseM" degree specification) which are mainly represented as strings in JSON or YAML formats and may not be allowed in other languages used to write these dictionaries to file. The easiest way is to save tuples as ```"(1, C)"``` (different from ```string(tuple(1, "C"))```) and use `parse_ace_basis_keys()` (also done within `fill_defaults()`) to parse that into ```(1, "C")```. 

```@docs
parse_ace_basis_keys
```

`db_params()` returns only those parameters needed to construct the least-squares database. 

```@docs
db_params
```
