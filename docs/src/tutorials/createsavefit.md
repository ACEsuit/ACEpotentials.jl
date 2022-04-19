# TiAl Tutorial

Make sure to set the environment variable `JULIA_NUM_THREADS` to enable threading in the least squares assembly `LsqDB()`. Also, BRR/ARD solvers will see increased performance if you enable Python numpy threading (most likely `MKL_NUM_THREADS`, can be confirmed by checking 
```python
import numpy; numpy.show_config()
```
in Python.

For fitting interatomic potentials using Julia the packages ACE1 and IPFitting are required.

```julia
using ACE1
using IPFitting
```

Reading in the `.xyz` containing the [TiAl configurations](https://raw.githubusercontent.com/ACEsuit/ACE1pack.jl/main/src/tutorials/tial/TiAl_tutorial_DB.xyz). `energy_key`, `force_key` and `virial_key` need to specified in order to make sure correct data is read. Slicing is done here to reduce fitting time such that it is feasible on a laptop. 

```julia
al = IPFitting.Data.read_xyz(@__DIR__() * "/TiAl_tutorial_DB.xyz", energy_key="energy", force_key="force", virial_key="virial")[1:10:end];
```

Next typical interatomic distance between the atoms `r0` is defined. Nearest neighbour distances of Ti and Al are quite similar (`rnn(:Ti)`=2.89607, `rnn(:Al)`=2.86378) so here we just take the average. For atomistic systems with large differences in neighour distances (or "size" of atoms) it is advised to use `MultiTransforms()` as defined in Tutorial `?` per element pair.  

```julia
r0 = 0.5*(rnn(:Ti) + rnn(:Al))
```

The rotational and permutation invariant ACE basis, `rpi_basis` is defined as follows. Here `N` is the correlation order, `r0` the previously defined typical nearest neighbour distance, `rin` the inner cutoff, `rcut` the outer cutoff and `maxdeg` the maximum polynomial degree of the ACE basis.

```julia
ACE_basis = rpi_basis(species = [:Ti, :Al],
                              N = 3,
                              r0 = r0,
                              rin = 0.6 * r0,
                              rcut = 5.5,
                              maxdeg = 6)
```

A simple pair-potential is defined below where `pin`=0 means no inner cutoff, and `pcut`=1

```jula
2B_basis = pair_basis(species = [:Ti, :Al],
      r0 = r0,
      maxdeg = 6,
      rcut = 7.0,
      pcut = 1,
      pin = 0)  
```

A `IPSuperBasis` is created by combining the pair-potential and the ACE basis. The pair-potential 

```julia
B = JuLIP.MLIPs.IPSuperBasis([Bpair, ACE_B]);
```

Using the combined basis and the configurations a least squares database is created. When `""` is specified the database is not saved to disk. If a filename `fname` is specified the least squares can be read in using `LsqDB("fname")` 

```julia
dB = LsqDB("", B, al)
```

A reference potential containing the isolated energies is defined using `OneBody()`. These energies will be subtracted from the total energy observations when fitting.
```julia
Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)
```

Weights are required per energy/force/virial observations and can be specified per configuration type. 
```julia
weights = Dict(
        "default" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ),
        "FLD_TiAl" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ))
```

Next a `solver` dictionary is created specifying the regularising method and its parameters.

## LSQR
```julia
solver = Dict(
        "solver" => :lsqr,
        "lsqr_damp" => 5e-3,
        "lsqr_atol" => 1e-6)
```
Iterative solver where `lsqr_damp` is the `L2` penalty and `lsqr_atol` the convergence parameter.

## RRQR
```julia
solver = Dict(
        "solver" => :rrqr,
        "rrqr_tol" => 1e-5)
```
Rank-revealing QR factorisation determines a low rank solution to the linear system. Smaller "rrqr_tol" means less regularisation. 

## Bayesian Ridge Regression (BRR)
```julia
solver = Dict(
        "solver" => :brr,
        "brr_tol" => 1e-3)
```        
Bayesian Ridge Regression performing evidence maximisation. `brr_tol` sets the convergence for the marginal log likelihood convergence, default is`1e-3`. 

## Automatic Relevance Determination (ARD)

```julia
solver= Dict(
         "solver" => :ard,
         "ard_tol" => 1e-3,
         "ard_threshold_lambda" => 10000)
```

Automatic Relevance Determination performing evidence maximisation. `ard_tol` sets the convergence for the marginal log likelihood convergence, default is`1e-3`. `ard_threshold_lambda` is the threshold for removing the basis functions with low relevance, default is `10000`.

Laplacian preconditioning can be used to penalise highly oscillatory basis function. This is done by creating the `P` matrix below and adding it to the `solver` dictionary.
```julia
using LinearAlgebra
rlap_scal = 3.0
P = Diagonal(vcat(ACE1.scaling.(dB.basis.BB, rlap_scal)...))
solver["P"] = P
```

Fitting is done using the `lsqfit` command, taking the least squares database `dB`, weights, reference potential and solver. Specifying `error_table=true` calculated the errors on the training database. The `lsqinfo` contains a wide range of properties of the potential, such as weights, solver and parameter information. 
```julia
IP, lsqinfo = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true);
```

Print the table containing the training database errors
```julia
rmse_table(lsqinfo["errors"])
```

Saving the potential is done as follows.
```julia
save_dict("./TiAl_tutorial_pot.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
```

Exporting to LAMMPs format (using the PACE calculator).
```julia
ACE1.ExportMulti.export_ACE("./TiAl_tutorial_pot.yace", IP)
```
