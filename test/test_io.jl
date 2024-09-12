
using Test, ACEpotentials, AtomsBuilder 
using AtomsCalculators: potential_energy
using Polynomials4ML.Testing: print_tf

model_spec = Dict("model_name" => "ACE1",
                  "elements" => ["Ti", "Al"],
                  "rcut" => 5.5,
                  "order" => 3,
                  "totaldegree" => 8)
model = ACEpotentials.make_model(model_spec)
set_parameters!(model, randn(length_basis(model)))

fname = tempname() * ".json" 
ACEpotentials.save_model(model, fname; model_spec = model_spec)

model1, meta = ACEpotentials.load_model(fname)

for ntest = 1:10 
    sys = rattle!(bulk(:Al, cubic=true) * 2, 0.1) 
    sys = randz!(sys, [:Ti => 0.5, :Al => 0.5]) 
    print_tf( @test potential_energy(sys, model) ≈ potential_energy(sys, model1) ) 
end

rm(fname)