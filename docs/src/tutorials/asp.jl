# # Basic Workflow
#
# This short tutorial introduces the use of the Lasso Homotopy (ASP) and Orthogonal Matching Pursuit (OMP) solvers.
# These are sparse solvers that compute the entire regularization path, 
# providing insight into how the support evolves as the regularization parameter changes.
# For more details on the algorithms and their implementation,
# see [ActiveSetPursuit.jl](https://github.com/MPF-Optimization-Laboratory/ActiveSetPursuit.jl)

# We start by importing `ACEpotentials` (and possibly other required libraries)

using Distributed, Random, SparseArrays 
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials, PrettyTables
using ACEpotentials.Models: fast_evaluator

##
sym = :Si

# Since the sparse solvers pick out the most relevant features for us, we typically start 
# with a model with a large basis.
# Here we use the `ace1_model` function to create a model with a total degree of 20 and polynomial order of 3.

model = ace1_model(elements = [sym,], order = 3, totaldegree = 23)
P = algebraic_smoothness_prior(model; p = 4)

# Next, we load a dataset. We split the dataset into training, validation, and test sets.
_train_data, test_data, _ = ACEpotentials.example_dataset("Zuo20_$sym")
shuffle!(_train_data); 
isplit = floor(Int, 0.8 * length(_train_data))
train_data = _train_data[1:isplit] 
val_data = _train_data[isplit+1:end]

# We can now assemble the linear system for the training and validation sets.
At, yt, Wt = ACEpotentials.assemble(train_data, model) 
Av, yv, Wv = ACEpotentials.assemble(val_data, model)

# We can now compute sparse solution paths using the `ASP` and `OMP` solvers.
# These solvers support customizable selection criteria for choosing a solution along the path.
#
# The `select` keyword controls which solution is returned:
# - `:final` selects the final iterate on the path.
# - `(:bysize, n)` selects the solution with exactly `n` active parameters.
# - `(:byerror, ε)` selects the smallest solution whose validation error is within a factor `ε` of the minimum validation error.

# The `tsvd` keyword controls whether the solution is post-processed using truncated SVD.
# This is often beneficial for `ASP`, as ℓ1-regularization can shrink coefficients toward zero too aggressively.

# The `actMax` keyword controls the maximum number of active parameters in the solution. 

solver_asp = ACEfit.ASP(; P = P, select = :final, tsvd = true, actMax = 1300,  loglevel = 1)
asp_result = ACEfit.solve(solver_asp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)


# We can also compute the OMP path, which is a greedy algorithm that selects the most relevant features iteratively.

solver_omp = ACEfit.OMP(; P = P, select = :final, tsvd = false, actMax = 1300, loglevel = 1)
omp_result = ACEfit.solve(solver_omp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)


# To demonstrate the use of the sparse solvers, we will generate models with different numbers of active parameters.
# We can select the final model, a model with 500 active parameters, and a model with a validation error within 1.3 times the minimum validation error.
# We can use the `ACEfit.asp_select` function to select the desired models from the result.

asp_final = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(asp_result, :final)[1])
asp_size_500  = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(asp_result, (:bysize, 500))[1])
asp_error13  = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(asp_result, (:byerror, 1.3))[1])

pot_final = fast_evaluator(asp_final; aa_static = false)  
pot_500 = fast_evaluator(asp_size_500; aa_static = true)
pot_13 = fast_evaluator(asp_error13; aa_static = true)

err_13 = ACEpotentials.compute_errors(test_data,  pot_13)
err_500 = ACEpotentials.compute_errors(test_data,  pot_500)
err_fin = ACEpotentials.compute_errors(test_data, pot_final)

header = ["", "Energy MAE (meV)", "Force MAE (eV/Å)"]

e_force_table = [
    "ASP(1.3)"  round(err_13["mae"]["set"]["E"] * 1000, digits=3)  round(err_13["mae"]["set"]["F"], digits=3)
    "ASP(500)"  round(err_500["mae"]["set"]["E"] * 1000, digits=3)  round(err_500["mae"]["set"]["F"], digits=3)
    "ASP(1300)" round(err_fin["mae"]["set"]["E"] * 1000, digits=3) round(err_fin["mae"]["set"]["F"], digits=3)
]

pretty_table(e_force_table; header = header)


# Similarly, we can compute the errors for the OMP models.

omp_final = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(omp_result, :final)[1])
omp_500  = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(omp_result, (:bysize, 500))[1])
omp_13  = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(omp_result, (:byerror, 1.3))[1])

pot_fin = fast_evaluator(omp_final; aa_static = false) 
pot_500 = fast_evaluator(omp_500; aa_static = true)
pot_13 = fast_evaluator(omp_13; aa_static = true)

err_13 = ACEpotentials.compute_errors(test_data,  pot_13)
err_500 = ACEpotentials.compute_errors(test_data,  pot_500)
err_fin = ACEpotentials.compute_errors(test_data, pot_fin)


e_force_table = [
    "OMP(1.3)"  round(err_13["mae"]["set"]["E"] * 1000, digits=3)  round(err_13["mae"]["set"]["F"], digits=3)
    "OMP(500)"  round(err_500["mae"]["set"]["E"] * 1000, digits=3)  round(err_500["mae"]["set"]["F"], digits=3)
    "OMP(1300)" round(err_fin["mae"]["set"]["E"] * 1000, digits=3) round(err_fin["mae"]["set"]["F"], digits=3)
]

pretty_table(e_force_table; header = header)

