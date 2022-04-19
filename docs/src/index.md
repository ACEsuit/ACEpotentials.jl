```@meta
CurrentModule = ACE1docs
```

# ACE1 User Documentation 

This is the documentation for a collection of codes for parameterising interatomic potentials in terms of the atomic cluster expansion, i.e., body-ordered invariant polynomials. The Julia packages that are adressed by this documentation are the following 

* `ACE1.jl` specifies the parameterisation of interatomic potentials in terms of the (linear) atomic cluster expansion; it provides functions to generate invariant basis sets, and to evaluate the resulting interatomic potentials.
* `IPFitting.jl` supplied the functionality for parameter estimation. It focuses purely on linear models and linear observations, but provides various tools to deal with the typical data to which interatomic potentials are fitted (total energies, forces, virials, etc) and the reading and transforming of training data. A broad range of solvers are available through this package. 
* `JuLIP.jl` is a simple molecular simulation code in pure Julia, focusing primarily on an infrastructure to develop interatomic potentials. It provides various generic functions on top of which all our packages on this page build.
* `ACE1pack.jl` has two purposes: (1) import and re-export `ACE1.jl, IPFitting.jl, JuLIP.jl` with guaranteed version compatibility; and (2) several convenience wrappers for `ACE1.jl` and `IPFitting.jl`
* `ACEinterfaces.jl` provides interfaces to use ACE potentials from other languages 


### Key references

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](doi:10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)


```@index
```
