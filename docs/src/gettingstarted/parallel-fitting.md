# Parallel Fitting

`ACEpotentials` and `ACEfit` may be accelerated with one or more parallelization options.

### Distributed fitting (multiple processes)

Some routines (particularly those that assemble the linear problem) make use of `Julia`'s multi-processing capabilities. These routines automatically utilize any available worker processes, which are initiated in one of two ways.

First, one may generate the workers when starting `Julia`. Setting `JULIA_PROJECT` beforehand is crucial in this case. The example starts `Julia` with seven additional worker processes (so, eight processes in total).
```bash
export JULIA_PROJECT=/path/to/project
julia --project=path/to/project -p 7
```

Alternatively, one may create workers directly within a `Julia` script. The `exeflags` argument to `addprocs` propogates project information, and the `@everywhere` macro is necessary to ensure all processes load the module. 
```julia
using Distributed
addprocs(7, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials
```

### Parallel `BLAS` or `LAPACK`

Many `ACEfit` solvers, and possibly other routines, utilize `BLAS` or `LAPACK`. To see benefits from threading, one should set one or more of the following environment variables, depending on the particular library used.
```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
```
