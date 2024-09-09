
# using Pkg; Pkg.activate(".");
# using TestEnv; TestEnv.activate();

using ACEpotentials, Test 
using Polynomials4ML.Testing: print_tf, println_slim

# there are no real tests for envelopes yet. The only thing we have is 
# a plot of the envelopes to inspect manually.

##
# this code block should normally be just commented out and is just intended 
# for a quick visual inspection of the transforms. 

#=
using Plots 
rcut = 6.5 
r0 = 2.3

trans_2_2 = ACEpotentials.Models.agnesi_transform(r0, rcut, 2, 2)
trans_2_4 = ACEpotentials.Models.agnesi_transform(r0, rcut, 2, 4)
trans_1_3 = ACEpotentials.Models.agnesi_transform(r0, rcut, 1, 3)

rr = range(-0.5, rcut+0.5, length=200)

y_2_2 = ACEpotentials.Models.evaluate.(Ref(trans_2_2), rr)
y_2_4 = ACEpotentials.Models.evaluate.(Ref(trans_2_4), rr)
y_1_3 = ACEpotentials.Models.evaluate.(Ref(trans_1_3), rr)
dy_2_2 = ACEpotentials.Models.evaluate_d.(Ref(trans_2_2), rr)
dy_2_4 = ACEpotentials.Models.evaluate_d.(Ref(trans_2_4), rr)
dy_1_3 = ACEpotentials.Models.evaluate_d.(Ref(trans_1_3), rr)

plt1 = plot(rr, y_2_2, label="Agnesi(2,2)", lw=2, legend=:topleft, ylims = (-1.2, 1.2))
plot!(rr, y_2_4, label="Agnesi(2,4)", lw=2)
plot!(rr, y_1_3, label="Agnesi(1,3)", lw=2)
vline!([0.0, r0, rcut], ls=:dash, lw=2, label="rin, r0, rcut")

plt2 = plot(rr, dy_2_2, label="∇Agnesi(2,2)", lw=2, legend=:topright, ylims = (-0.05, 0.8))
plot!(rr, dy_2_4, label="∇Agnesi(2,4)", lw=2)
plot!(rr, dy_1_3, label="∇Agnesi(1,3)", lw=2)
vline!([0.0, r0, rcut], ls=:dash, lw=2, label="rin, r0, rcut")

plot(plt1, plt2, layout=(2,1), size = (600, 800))
=#

##

@info("Testing agnesi transforms")
rcut = 6.5 
r0 = 2.3

trans_2_2 = ACEpotentials.Models.agnesi_transform(r0, rcut, 2, 2)
trans_2_4 = ACEpotentials.Models.agnesi_transform(r0, rcut, 2, 4)
trans_1_3 = ACEpotentials.Models.agnesi_transform(r0, rcut, 1, 3)

for trans in [trans_2_2, trans_2_4, trans_1_3] 
   println_slim( @test ACEpotentials.Models.test_normalized_transform(trans_2_2) )
end

