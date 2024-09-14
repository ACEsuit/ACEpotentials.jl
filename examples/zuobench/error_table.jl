# This script reproduces the error table from the 2023/24 ACEpotentials 
# paper, but with the new version 0.8 ACE models.

using Distributed, PrettyTables
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

# the dataset is provided via ACE1pack artifacts as a convenient benchmarkset
# the following chemical symbols are available:
syms = [:Ni, :Cu, :Li, :Mo, :Si, :Ge]

totaldegree_tiny = [ 16, 12, 8 ]   # very small model: ~ 120  basis functions
totaldegree_sm = [ 20, 16, 12 ]   # small model: ~ 300  basis functions
totaldegree_lge = [ 25, 21, 17 ]  # large model: ~ 1000 basis functions              

## 

err = Dict("lge" => Dict("E" => Dict(), "F" => Dict()), 
            "sm" => Dict("E" => Dict(), "F" => Dict()) )

for sym in syms 
   @info("---------- fitting $(sym) ----------")
   train, test, _ = ACEpotentials.example_dataset("Zuo20_$sym")

   # specify the models 
   model_sm = ace1_model(elements = [sym,], order = 3, totaldegree = totaldegree_sm)
   model_lge = ace1_model(elements = [sym,], order = 3, totaldegree = totaldegree_lge)
   @info("$sym models: length = $(length_basis(model_lge)), $(length_basis(model_sm))")

   # train the model 
   solver = ACEfit.BLR(; factorization = :svd)
   acefit!(train, model_sm;  solver=solver); GC.gc()
   acefit!(train, model_lge; solver=solver); GC.gc() 

   # compute and store errors for later visualisation
   err_sm  = ACEpotentials.linear_errors(test, model_sm)
   err_lge = ACEpotentials.linear_errors(test, model_lge)
   err["sm" ]["E"][sym] =  err_sm["mae"]["set"]["E"] * 1000
   err["sm" ]["F"][sym] =  err_sm["mae"]["set"]["F"]
   err["lge"]["E"][sym] = err_lge["mae"]["set"]["E"] * 1000
   err["lge"]["F"][sym] = err_lge["mae"]["set"]["F"]
end


##

header = ([ "", "ACE(sm)", "ACE(lge)", "GAP", "MTP"])
e_table_gap_mtp = [ 0.42  0.48; 0.46  0.41; 0.49  0.49; 2.24  2.83; 2.91  2.21; 2.06  1.79]
f_table_gap_mtp = [ 0.02  0.01; 0.01  0.01; 0.01  0.01; 0.09  0.09; 0.07  0.06; 0.05  0.05]
      
e_table = hcat(string.(syms),
         [round(err["sm"]["E"][sym], digits=3) for sym in syms], 
         [round(err["lge"]["E"][sym], digits=3) for sym in syms],
         e_table_gap_mtp)
     
f_table = hcat(string.(syms),
         [round(err["sm"]["F"][sym], digits=3) for sym in syms], 
         [round(err["lge"]["F"][sym], digits=3) for sym in syms],
         f_table_gap_mtp)         

println("Energy Error")         
pretty_table(e_table; header = header)

println("Force Error")         
pretty_table(f_table; header = header)

##

# pretty_table(e_table, backend = Val(:latex), label = "Energy MAE", header = header)
# pretty_table(f_table, backend = Val(:latex), label = "Forces MAE", header = header)