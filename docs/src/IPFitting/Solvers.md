# Least squares solvers

### `:lsqr`

The `:lsqr` solver solves the linear system with ``l^2`` regularisation:

```math
\mathbf{c} = \text{arg} \min_\mathbf{c} \| \mathbf{y} - \Psi \mathbf{c} \|^2 + \lambda^2 \| \mathbf{c} \|^2
```

The solver dictionary should have the following arguments:
```julia
solver = Dict(
        "solver" => :lsqr,
        "lsqr_damp" => 5e-3,
        "lsqr_atol" => 1e-6)
```
where `lsqr_damp` is ``\lambda`` in the equation above. The implementation is iterative and `lsqr_atol` is a convergence tolerance at which to stop the alogrithm.

### `:rrqr`

Rank-revealing QR factorisation determines a low rank solution to the linear system. Smaller "rrqr_tol" means less regularisation. 

```julia
solver = Dict(
        "solver" => :rrqr,
        "rrqr_tol" => 1e-5)
```

### `:brr`

The `:brr` - Bayesian Ridge Regression - is a wrapper for `scikit learn`'s BayseianRidge linear model [see here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html). A Gaussian prior on the parameter vector, and a Gaussian likelihood function are used to copmute the maximum posterior probability parameter vector. Following this, the hyperparamters of the Gaussian priors are optimised by maximising the marginal log likelihood of the obsrevations. 

```julia
solver = Dict(
        "solver" => :brr, 
        "brr_tol" => 1e-3)
```

## Automatic Relevance Determination (ARD)

```julia
solver= Dict(
         "solver" => :ard,
         "ard_tol" => 1e-3,
         "ard_threshold_lambda" => 10000)
```

Automatic Relevance Determination performing evidence maximisation. `ard_tol` sets the convergence for the marginal log likelihood convergence, default is`1e-3`. `ard_threshold_lambda` is the threshold for removing the basis functions with low relevance, default is `10000`.






