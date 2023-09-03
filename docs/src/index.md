```@meta
CurrentModule = ACEpotentials
```

# ACEpotentials.jl Documentation 

`ACEpotentials.jl` provides a user-oriented interface to a collection of Julia packages facilitating the generation and use of atomic cluster expansion (ACE) interatomic potentials. ACE models are defined in terms of body-ordered invariant features of atomic environments. For details we refer to [our brief introduction](gettingstarted/aceintro.md) and to the references listed below. 

For a quick start, we recommend reading the installation instructions, followed by the tutorials. 


### Overview 

`ACEpotentials.jl` is a user-oriented tool for producing ACE potentials. It relies on several more targeted packages implementing different aspects of ACE modelling and fitting, `namely, `JuLIP.jl`, `ACE1.jl`, `ACE1x.jl`, `ACEfit.jl`, and `ACEmd.jl`. `ACEpotentials` re-exports the features of these packages, ensuring version compatibility, while also providing additional fitting and analysis routines. These pages document `ACEpotentials`, as well as the most relevant features of the packages it depends on.

A short summary of packages behind `ACEpotentials`:

* `JuLIP.jl` is a simple molecular simulation code in pure Julia, focusing primarily on an infrastructure to develop interatomic potentials. It provides various generic functions on top of which all our packages on this page build. 
* `ACE1.jl` specifies the parameterisation of interatomic potentials in terms of the (linear) atomic cluster expansion; it provides functions to generate invariant basis sets, and to evaluate the resulting interatomic potentials. 
* `ACE1x.jl` is an extension of `ACE1.jl` incorporating new experimental features. We expect to merge these packages over time.
* `ACEfit.jl` supplies the functionality for parameter estimation. Presently, it focuses purely on linear models and linear observations. ACEpotentials provides various tools to deal with the typical data to which interatomic potentials are fitted (total energies, forces, virials, etc) and the reading and transforming of training data. A broad range of solvers are available through this ACEfit. 
* `ACEmd.jl` is a new implementation of ACE calculators compatible with `Molly.jl`. 

!!! warning "JSON interface"
    `ACEpotentials.jl` contains JSON and command line interfaces that can be used to fit ACE potentials without needing to write Julia scripts. These have not been updated to include the most recent advances in ACE models available through the Julia interfaces. Until this changes, we recommend using `ACEpotentials.jl` only through the Julia interfaces. If this is a feature important to you, please file an issue or bump an existing issue to accelerate us updating this feature.

```@raw html
<!---
### Three Ways to Work with `ACEpotentials.jl`
1. Via Julia scripts: 
   - `ACE1.jl` can be used directly for maximal fine-grained control over the model parameter. Some extensions are implemented in `ACE1x.jl`. 
   - `ACE1x.jl` also provides an intermediate convenience layer with many default parameters that have proven successful for a range of applications.
2. JSON interface: `ACEpotentials.jl` provides wrapper functions to generate ACE models from dictionaries of model parameters. **WARNING:** This JSON interface will not always be up to date with the latest modeling options provided in `ACE1x.jl`.
3. From the Command line: `ACEpotentials` provides scripts to fit potentials from the command line. Fitting is controlled by a dictionary of parameters in a `.json` or `.yaml` file to specify the model and fitting options. Only functionality supported by the JSON interface is available. 
--->
```

### References

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)
