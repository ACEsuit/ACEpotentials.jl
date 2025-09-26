
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://acesuit.github.io/ACEpotentials.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://acesuit.github.io/ACEpotentials.jl/stable)
[![Build Status](https://github.com/acesuit/ACEpotentials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/acesuit/ACEpotentials.jl/actions/workflows/CI.yml?query=branch%3Amain)

![alt text](logo/acesuit_E_sm.png)

`ACEpotentials.jl` facilitates the creation and use of atomic cluster expansion (ACE) interatomic potentials. See [the documentation](https://acesuit.github.io/ACEpotentials.jl/dev) for installation instructions, tutorials, and more.

## Notes on Versions

- Version 0.6.x uses `ACE1.jl` as a backend. It is mature and suitable for linear models with few species. This is not longer actively developed, but critical bugfixes can still be provided.  [[docs-v0.6]](https://acesuit.github.io/ACEpotentials.jl/v0.6/)
- Version 0.7.x is reserved
- Version 0.8.x and onwards provides a new and much more flexible implementation, and integrates with the [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) ecosystem. Most but not all features from 0.6.x have been ported to this re-implementation. Usability should be the same or improved for most end-users. For developers this provides a much more flexible framework for experimentation. [[docs-v0.8]](https://acesuit.github.io/ACEpotentials.jl/dev/)

## Contributing

Contributions are very welcome. Until clear guidelines and practices are established, we recommend to open an issue where the bugfix or enhancement can be discussed, before starting a pull request. We will do our best to respond in a timely manner.

## Quick Start 

- Install Julia 1.11
- Create new folder a.g. `acetutorial`; Open a shell
- Create a new project in `acetutorial` and install `ACEpotentials.jl`
```
julia --project=. 
] 
registry add https://github.com/ACEsuit/ACEregistry
add ACEpotentials
```
- Install the Julia tutorials (this installs two Jupyter notebook tutorials)
```julia-repl
using ACEpotentials
ACEpotentials.copy_tutorial()
```
- Work through the tutorials.

