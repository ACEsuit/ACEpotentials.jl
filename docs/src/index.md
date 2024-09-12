```@meta
CurrentModule = ACEpotentials
```

# ACEpotentials.jl Documentation 

`ACEpotentials.jl` facilitates the creation and use of atomic cluster expansion (ACE) interatomic potentials. For a quick start, we recommend reading the installation instructions, followed by the tutorials. 

ACE models are defined in terms of body-ordered invariant features of atomic environments. For mathematical details, see [this brief introduction](gettingstarted/aceintro.md) and the references listed below.


### Overview 

`ACEpotentials.jl` ties together several Julia packages implementing different aspects of ACE modelling and fitting and provides some additional fitting and analysis tools for convenience. For example, it provides routines for parsing and manipulating the data to which interatomic potentials are fit (total energies, forces, virials, etc). Moreover, it integrates ACE potentials with the [JuliaMolSim](https://github.com/JuliaMolSim) eco-system. These pages document `ACEpotentials`together with the relevant parts of the wider ecosystem.

### References

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[DOI]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) [[arxiv]](https://arxiv.org/abs/2003.00221)

* G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454, 110946, 2022. [[DOI]](https://doi.org/10.1016/j.jcp.2022.110946) [[arxiv]](https://arxiv.org/abs/1911.03550)

* W. C. Witt, C. van der Oord, E. Gelžinyté, T. Järvinen, A. Ross, J. P. Darby, C. H. Ho, W. J. Baldwin, M. Sachs, J. Kermode, N. Bernstein, G. Csányi, and C. Ortner. ACEpotentials.jl: A Julia Implementation of the Atomic Cluster Expansion. J. Chem. Phys., 159:164101, 2023. [[DOI]](https://doi.org/10.1063/5.0158783) [[arxiv]](https://arxiv.org/abs/2309.03161)

### Key Dependencies

* [`Polynomials4ML.jl`](https://github.com/ACEsuit/Polynomials4ML.jl) : basic kernels for embeddings and tensors
* [`EquivariantModels.jl`](https://github.com/ACEsuit/EquivariantModels.jl) : tools for equivariant model building
* [`RepLieGroups.jl`](https://github.com/ACEsuit/RepLieGroups.jl) : coupling coefficients for equivariant tensors
* [`ACEfit.jl`](https://github.com/ACEsuit/ACEfit.jl) : unified interface to various regression algorithms
* [`AtomsBase.jl`](https://github.com/JuliaMolSim/AtomsBase.jl) : community interface for atomic structures / systems
* [`AtomsCalculators.jl`](https://github.com/JuliaMolSim/AtomsCalculators.jl) : community interface for computing properties of systems
* [`ExtXYZ.jl`](https://github.com/libAtoms/ExtXYZ.jl) : reading and writing extended xyz format


### Useful Related Packages

* [`Molly.jl`](https://github.com/JuliaMolSim/Molly.jl) : main molecular dynamics package in Julia
* [`DFTK.jl`](https://github.com/JuliaMolSim/DFTK.jl) : pure Julia plane wave DFT code
* [`GeometryOptimization.jl`](https://github.com/JuliaMolSim/GeometryOptimization.jl) and [`GeomOpt.jl`](https://github.com/ACEsuit/GeomOpt.jl) 
