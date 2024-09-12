
using Distributed 
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials, PrettyTables

# the dataset is provided via ACE1pack artifacts as a convenient benchmarkset
# the following chemical symbols are available:
# syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]

# this element is quite interesting because it gives odd results 
sym = :Ni  # :Mo 

# start with a large-ish model 
totaldegree = [ 28, 23, 18 ] 
model = ace1_model(elements = [sym,], order = 3, totaldegree = totaldegree)
P = algebraic_smoothness_prior(model; p = 4)

@info("$sym model, basis length = $(length_basis(model))")

##

@info("---------- Assemble Training and Validation Systems ----------")
_train_data, test_data, _ = ACEpotentials.example_dataset("Zuo20_$sym")
Random.shuffle!(_train_data)
train_data = _train_data[1:5:end]
val_data = _train_data[2:5:end]
# datakeys = (:)
At, yt, Wt = ACEpotentials.assemble(train_data, model) 
Av, yv, Wv = ACEpotentials.assemble(val_data, model)

@info("Compute ASP Path")
solver = ACEfit.ASP(; P = P, select = (:byerror, 1.0), tsvd = true )
asp_result = ACEfit.solve(solver, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)

## 

@info("Look at the best model")
asp_result["C"]
set_parameters!(model, asp_result["C"])
ACEpotentials.linear_errors(test_data, model)




# 120 
# 300 
# 1000 

##

# header = ([ "", "ACE(sm)", "ACE(lge)", "GAP", "MTP"])
# e_table_gap_mtp = [ 0.42  0.48; 0.46  0.41; 0.49  0.49; 2.24  2.83; 2.91  2.21; 2.06  1.79]
# f_table_gap_mtp = [ 0.02  0.01; 0.01  0.01; 0.01  0.01; 0.09  0.09; 0.07  0.06; 0.05  0.05]
      
# e_table = hcat(string.(syms),
#          [round(err["sm"]["E"][sym], digits=3) for sym in syms], 
#          [round(err["lge"]["E"][sym], digits=3) for sym in syms],
#          e_table_gap_mtp)
     
# f_table = hcat(string.(syms),
#          [round(err["sm"]["F"][sym], digits=3) for sym in syms], 
#          [round(err["lge"]["F"][sym], digits=3) for sym in syms],
#          f_table_gap_mtp)         

# println("Energy Error")         
# pretty_table(e_table; header = header)

# println("Force Error")         
# pretty_table(f_table; header = header)

# ##

# pretty_table(e_table, backend = Val(:latex), label = "Energy MAE", header = header)
# pretty_table(f_table, backend = Val(:latex), label = "Forces MAE", header = header)