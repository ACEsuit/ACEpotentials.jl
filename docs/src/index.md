```@meta
CurrentModule = ACE1pack
```

# ACE1 and ACE1pack Documentation 

`ACE1.jl` is a Julia package for parameterising interatomic potentials in terms of the atomic cluster expansion, i.e., body-ordered invariant polynomials. `ACE1pack.jl` provides a user-oriented and compatibility layer. These pages document `ACE1pack` and to some limited extent also the packages it depend on.

### Three Ways to Work with `ACE1.jl`

1. Fitting from the Command line.

`ACE1pack` provides scripts to fit potentials from the command line. Fitting is controlled by a dictionary of parameters in a `.json` file to specify the model and fitting options.

2. Using the `ACE1pack` wrapper functions in julia

`ACE1pack` also provides helper functions which wrap the functionality of `ACE1.jl` in julia. 

3. Using `ACE1.jl` in julia directly. 

Finally, it is possible to use the ACE1 julia code directly.

### References

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)
