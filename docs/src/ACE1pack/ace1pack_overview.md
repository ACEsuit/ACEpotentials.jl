
# Overview

`ACE1pack.jl` has two purposes: (1) to import and re-export `ACE1.jl`, `IPFitting.jl`, `JuLIP.jl` with guaranteed version compatibility; and (2) to have several convenience wrappers for setting up the least-squares problem (`ACE1.jl` & `JuLIP.jl`) and solving it (`IPFitting.jl`). 

## General structure

The main convenience functions are:

* `make_ace_db()` - make the design matrix ("ACE database") for the least-squares problem. 
* `fit_ace_db()` - fit the given ACE database
* `fit_ace()` - `make_ace_db()` and `fit_ace_db()` in one go.

See [Fitting ACE](fit.md) for more information. 

All of these functions take nested dictionaries that specify various parameters in making ACE. For convenience, there are a number of `*params` functions exported by ACE1pack that return these dictionaries with complete set of parameters specified. These are: 

* `fit_params()` - highest-level parameters' dictionary, compatible with all of `make_ace_db()`, `fit_ace_db()` and `fit_ace()`, see [fit.md];
* `basis_params()` - parameters for constructing various bases for the design matrix of the ACE least squares database, see [Fitting ACE](fit.md);
* `degree_params()` - for specifying the degree of ACE basis, see [Constructing Basis](basis.md);
* `transform_params()` - parameters to specify the transform for a given basis, see [Constructing Basis](basis.md);
* `data_params()` - for reading geometries and to-be-fitted property values, see [Handling Data](data.md); 
* `regularizer_params()` - to set up an extra regularizer, see [Regularizers](regularizer.md);
* `solver_params()` - to set up solver for the least-squares problem, see [Solvers](solver.md). 

In addition, there are some utility functions:  

* `save_fit()` - save given potential to file, see [Fitting ACE](fit.md);
* `fill_defaults()` - recursively fills in default values for any of the optional parameters that were left unspecified, see [Helper Functions](helpers.md);
* `parse_ace_basis_keys()` - for parsing `"(element1, number)" -> ("element1", number)`-type entries that were read in from `.json` or `.yaml` files, see [Helper Functions](helpers.md); 
* `db_params()` - a subset of parameters returned by `fit_params()`, compatible with `make_ace_db()` only, see [Fitting ACE](fit.md); 
* `load_dict()` - reads in parameters from `.yaml` or `.json` format, see [Helper Functions](helpers.md).



TODO
* functions with different calls (fit_ace_db) have docs look correctly?
