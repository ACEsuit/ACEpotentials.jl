
@info("Some consistency tests for model interface")

# I'd like to start putting some simple consistency tests in here.
using Test, ACEpotentials
using ACE1.Testing: println_slim

N = 4
species = :Si
r0 = 2.3;  rcut = 5.5; rin = 0.65 * r0
rcut2 = 6.2; maxdeg2 = 4

Dn = Dict("default" => 1.0)
Dl = Dict("default" => 1.5)
# degrees for different correlation orders
Dd = Dict(1 => 20, 2 => 20, 3 => 15, 4 => 10)   
DEG = ACE1.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

model1 = acemodel(species = species, N = N,
                    maxdeg = 1, D = DEG, 
                    r0 = r0, rcut = 5.5, rin = rin, 
                    rcut2 = rcut2, maxdeg2 = maxdeg2);

model2 = acemodel(species = species, N = N,
                  wL = Dl["default"], 
                  maxdeg = [Dd[n] for n = 1:N],
                  r0 = r0, rcut = 5.5, rin = rin, 
                  rcut2 = rcut2, maxdeg2 = maxdeg2);
                  
println_slim(@test model1.basis == model2.basis)

