```@meta
CurrentModule = ACEpotentials
```

# ACEpotentials.jl Documentation 

`ACEpotentials.jl` provides a user-oriented and interface to a collection of several Julia packages that interoperate to fit models for atomic cluster expansion (ACE) interatomic potentials. ACE models are defined in terms of body-ordered polynomial invariant features of atomic environment. For details we refer to [our brief introduction](gettingstarted/aceintro.md) and to the references listed below. 

For a quick start, we recommend reading the installation instructions, followed by the tutorials. 


!!! warning "JSON interface"
   `ACEpotentials.jl` contains JSON and command line interfaces that can be used to fit ACE potentials without needing to write Julia scripts. These have not been updated to include the most recent advances in ACE models available through the Julia interfaces. Until this changes, we recommend using `ACEpotentials.jl` only through the Julia interfaces. If this is a feature important to you, please file an issue or bump an existing issue to accelerate us updating this feature.

<!---
### Three Ways to Work with `ACEpotentials.jl`
1. Via Julia scripts: 
   - `ACE1.jl` can be used directly for maximal fine-grained control over the model parameter. Some extensions are implemented in `ACE1x.jl`. 
   - `ACE1x.jl` also provides an intermediate convenience layer with many default parameters that have proven successful for a range of applications.
2. JSON interface: `ACEpotentials.jl` provides wrapper functions to generate ACE models from dictionaries of model parameters. **WARNING:** This JSON interface will not always be up to date with the latest modeling options provided in `ACE1x.jl`.
3. From the Command line: `ACEpotentials` provides scripts to fit potentials from the command line. Fitting is controlled by a dictionary of parameters in a `.json` or `.yaml` file to specify the model and fitting options. Only functionality supported by the JSON interface is available. 
--->

### References

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)

### Detailed Overview 

`ACEpotentials.jl` has two purposes: (1) to import and re-export `ACE1.jl`, `ACE1x.jl`, `ACEfit.jl`, `JuLIP.jl`, `ACEmd.jl` with guaranteed version compatibility; and (2) to have several convenience wrappers for setting up the least-squares problem and solving it.

`ACE1.jl` and `ACE1x.jl` are Julia packages for parameterising interatomic potentials in terms of the atomic cluster expansion, i.e., body-ordered invariant polynomials. 
`ACEpotentials.jl` provides a user-oriented and convenience and compatibility layer. 
These pages document `ACEpotentials` and to some limited extent also the packages it depends on.

A short summary of packages behind `ACEpotentials`:

* `ACE1.jl` specifies the parameterisation of interatomic potentials in terms of the (linear) atomic cluster expansion; it provides functions to generate invariant basis sets, and to evaluate the resulting interatomic potentials. `ACE1x.jl` is an extension of `ACE1.jl` incorporating new experimental features. We expect to merge these packages over time.
* `ACEfit.jl` supplies the functionality for parameter estimation. Presently, it focuses purely on linear models and linear observations. ACEpotentials provides various tools to deal with the typical data to which interatomic potentials are fitted (total energies, forces, virials, etc) and the reading and transforming of training data. A broad range of solvers are available through this ACEfit. 
* `JuLIP.jl` is a simple molecular simulation code in pure Julia, focusing primarily on an infrastructure to develop interatomic potentials. It provides various generic functions on top of which all our packages on this page build. 
* `ACEmd.jl` is a new implementation of ACE calculators compatible with `Molly.jl`. 
