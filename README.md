# ACEpotentials.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://acesuit.github.io/ACEpotentials.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://acesuit.github.io/ACEpotentials.jl/stable)
[![Build Status](https://github.com/acesuit/ACEpotentials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/acesuit/ACEpotentials.jl/actions/workflows/CI.yml?query=branch%3Amain)

`ACEpotentials.jl` facilitates the creation and use of atomic cluster expansion (ACE) interatomic potentials. See [the documentation](https://acesuit.github.io/ACEpotentials.jl/stable) for installation instructions, tutorials, and more.

## Notes on Versions

- Version 0.6.x uses `ACE1.jl` as a backend. It is mature and suitable for linear models with few species. This is not longer actively developed, but critical bugfixes can still be provided. 
- Version 0.7.x is reserved
- Version 0.8.x and onwards provides a new and much more flexible implementation, and integrates with the [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) ecosystem. Most but not all features from 0.6.x have been ported to this re-implementation. Usability should be the same or improved for most end-users. For developers this provides a much more flexible framework for experimentation.

