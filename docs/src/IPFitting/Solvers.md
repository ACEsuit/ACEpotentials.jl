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
        "damp" => 5e-3,
        "atol" => 1e-6)
```
where `damp` is ``\lambda`` in the equation above. The implementation is iterative and `atol` is a convergence tolerance at which to stop the alogrithm.

### `:rrqr`

```julia
solver = Dict(
        "solver" => :rrqr,
        "rrqr_tol" => 1e-5)
```

### `:brr`

The `:brr` - Bayesian Ridge Regression - is a wrapper for `scikit learn`'s BayseianRidge linear model [see here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html). A Gaussian prior on the parameter vector, and a Gaussian likelihood function are used to copmute the maximum posterior probability parameter vector. Following this, the hyperparamters of the Gaussian priors are optimised by maximising the marginal log likelihood of the obsrevations. 

```julia
solver = Dict(
        "solver" => :brr)
```

Since this algorithm includes hyperparameter maximisation, no parameters are required in the solver dictionary.

```julia
solver= Dict(
         "solver" => :ard,
         "ard_tol" => 1e-4,
         "threshold_lambda" => 1e-2)
```