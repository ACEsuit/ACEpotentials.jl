```@meta
CurrentModule = ACEpotentials
```

# ACE1 and ACEpotentials Documentation 

`ACE1.jl` and `ACE1x.jl` are Julia packages for parameterising interatomic potentials in terms of the atomic cluster expansion, i.e., body-ordered invariant polynomials. 
`ACEpotentials.jl` provides a user-oriented and convenience and compatibility layer. 
These pages document `ACEpotentials` and to some limited extent also the packages it depends on.

### Three Ways to Work with `ACE1.jl`


1. Via Julia scripts: 
   - `ACE1.jl` can be used directly for maximal fine-grained control over the model parameter. Some extensions are implemented in `ACE1x.jl`. 
   - `ACE1x.jl` also provides an intermediate convenience layer with many default parameters that have proven successful for a range of applications.
2. JSON interface: `ACEpotentials` provides wrapper functions to generate ACE models from dictionaries of model parameters. **WARNING:** This JSON interface will not always be up to date with the latest modeling options provided in `ACE1x.jl`.
3. From the Command line: `ACEpotentials` provides scripts to fit potentials from the command line. Fitting is controlled by a dictionary of parameters in a `.json` or `.yaml` file to specify the model and fitting options. Only functionality supported by the JSON interface is available. 


### References

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)
