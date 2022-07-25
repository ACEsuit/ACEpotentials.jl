# Solvers

LSQR, RRQR, BRR & ARD solvers that solve the `Ax=b` problem with some regularisation are available via [IPFitting module](../IPFitting/Solvers.md). `solver_params()` is the way these are define and selected in ACE1pack. 

```@docs
solver_params
```

Additional regularizers (currently only "laplacian") are given via `regularizer_params()`

```@docs
regularizer_params
```
