
using Distributed, Random, SparseArrays 
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials, PrettyTables
using ACEpotentials.Models: fast_evaluator

##

# the dataset is provided via ACE1pack artifacts as a convenient benchmarkset
# the following chemical symbols are available:
syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]

# this element is quite interesting because it gives odd results 
sym = :Mo 

# start with a large-ish model 
@info("Generate model with ca 2000 basis functions")
totaldegree = [ 28, 24, 20 ]  # ≈ ca 2000 basis functions to select from 
model = ace1_model(elements = [sym,], order = 3, totaldegree = totaldegree)
P = algebraic_smoothness_prior(model; p = 4)

@info("$sym model, basis length = $(length_basis(model))")

##

@info("---------- Assemble Training and Validation Systems ----------")
_train_data, test_data, _ = ACEpotentials.example_dataset("Zuo20_$sym")
shuffle!(_train_data); 
isplit = floor(Int, 0.8 * length(_train_data))
train_data = _train_data[1:isplit] 
val_data = _train_data[isplit+1:end]

At, yt, Wt = ACEpotentials.assemble(train_data, model) 
Av, yv, Wv = ACEpotentials.assemble(val_data, model)

@info("Compute ASP Path")
solver = ACEfit.ASP(; P = P, select = :final, tsvd = true, actMax = 1000 )
asp_result = ACEfit.solve(solver, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)

## 

@info("Pick solutions for 100, 300, 1000 parameters, compute errors")

# select models from the model path 
model_1000 = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(asp_result, :final)[1])
model_300  = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(asp_result, (:bysize, 300))[1])
model_100  = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(asp_result, (:bysize, 100))[1])

# generate sparsified, faster evaluators 
pot_1000 = fast_evaluator(model_1000; aa_static = false) # static can cause stack overflow
pot_300 = fast_evaluator(model_300; aa_static = true)
pot_100 = fast_evaluator(model_100; aa_static = true)

@info("Evaluate errors on the test set") 
err_100 = ACEpotentials.compute_errors(test_data,  pot_100)
err_300 = ACEpotentials.compute_errors(test_data,  pot_300)
err_1000 = ACEpotentials.compute_errors(test_data, pot_1000)

##

header = ([ "", "ACE(100)", "ACE(300)", "ACE(1000)", "GAP", "MTP"])
e_table_gap_mtp = [ 0.42  0.48; 0.46  0.41; 0.49  0.49; 2.24  2.83; 2.91  2.21; 2.06  1.79]
f_table_gap_mtp = [ 0.02  0.01; 0.01  0.01; 0.01  0.01; 0.09  0.09; 0.07  0.06; 0.05  0.05]
i_sym = findfirst(syms .== sym)

e_table = hcat( [ string(sym) ],
            [ round( err_100["mae"]["set"]["E"] * 1000, digits=3), ], 
            [ round( err_300["mae"]["set"]["E"] * 1000, digits=3), ], 
            [ round(err_1000["mae"]["set"]["E"] * 1000, digits=3), ], 
            e_table_gap_mtp[i_sym:i_sym, :] )
     
f_table = hcat( [ string(sym) ],
            [ round( err_100["mae"]["set"]["F"], digits=3), ], 
            [ round( err_300["mae"]["set"]["F"], digits=3), ], 
            [ round(err_1000["mae"]["set"]["F"], digits=3), ], 
            f_table_gap_mtp[i_sym:i_sym, :] )


println("Energy Error (MAE)")
pretty_table(e_table; header = header)

println("Force Error (MAE)")
pretty_table(f_table; header = header)

