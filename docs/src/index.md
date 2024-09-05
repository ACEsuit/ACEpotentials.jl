```@meta
CurrentModule = ACEpotentials
```

# ACEpotentials.jl Documentation 

`ACEpotentials.jl` facilitates the creation and use of atomic cluster expansion (ACE) interatomic potentials. For a quick start, we recommend reading the installation instructions, followed by the tutorials. 

ACE models are defined in terms of body-ordered invariant features of atomic environments. For mathematical details, see [this brief introduction](gettingstarted/aceintro.md) and the references listed below.


### Overview 

`ACEpotentials.jl` ties together several Julia packages implementing different aspects of ACE modelling and fitting (e.g., `JuLIP.jl`, `ACE1.jl`, `ACE1x.jl`, `ACEfit.jl`, and `ACEmd.jl`). `ACEpotentials` re-exports their features, ensuring version compatibility, and provides additional fitting and analysis tools. For example, it provides routines for parsing and manipulating the data to which interatomic potentials are fit (total energies, forces, virials, etc). These pages document `ACEpotentials`together with the relevant parts of the wider ecosystem.

* `JuLIP.jl` is simple pure-Julia molecular simulation package that provides infrastructure for interatomic potentials. It is the foundation on which the other packages build.
* `ACE1.jl` parameterizes interatomic potentials in terms of the (linear) atomic cluster expansion. It provides generate invariant basis sets and functions that evaluate the resulting interatomic potentials.
* `ACE1x.jl` is an extension of `ACE1.jl` incorporating new experimental features. We expect to merge these packages over time.
* `ACEfit.jl` supplies the functionality for parameter estimation. Presently, it focuses purely on linear models and linear observations. A broad range of solvers are available. 
* `ACEmd.jl` is a new implementation of ACE calculators compatible with `Molly.jl`.

!!! warning "JSON and command line interfaces"
    An earlier version of `ACEpotentials.jl` contained JSON and command line interfaces that can be used to fit ACE potentials without needing to write Julia scripts. These have not been updated to include the most recent features available from the Julia interfaces. Until this changes, we recommend using `ACEpotentials.jl` only through the Julia interfaces. If this is a feature important to you, please file an issue or bump an existing issue to accelerate us updating this feature.

### References

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)

* William C. Witt, Cas van der Oord, Elena Gelžinyté, Teemu Järvinen, Andres Ross, James P. Darby, Cheuk Hin Ho, William J. Baldwin, Matthias Sachs, James Kermode, Noam Bernstein, Gábor Csányi, and Christoph Ortner. ACEpotentials.jl: A Julia Implementation of the Atomic Cluster Expansion. J. Chem. Phys., 159:164101, 2023. [[DOI]](https://doi.org/10.1063/5.0158783) [[arxiv]](https://arxiv.org/abs/2309.03161)

