"""
Debug script to investigate basis function ordering between Julia 1.11 and 1.12
"""

using ACEpotentials
using Random: seed!

println("Julia version: ", VERSION)
println()

# Create the same model as in test_bugs.jl
model = ACEpotentials.ACE1compat.ace1_model(
    elements = [:Ti, ],
    order = 3,
    totaldegree = 10,
    rcut = 6.0,
    Eref = [:Ti => -1586.0195, ])

seed!(1234)
params = randn(ACEpotentials.length_basis(model))
ACEpotentials.Models.set_parameters!(model, params)

# Print basis length
println("Basis length: ", ACEpotentials.length_basis(model))
println()

# Print first 20 parameter values
println("First 20 parameter values:")
println(model.ps.WB[1:20, 1])
println()

# Try to inspect the A spec
try
    a_spec = ACEpotentials.Models.get_nnll_spec(model.model.tensor)
    println("A spec length: ", length(a_spec))
    println("First 10 A spec entries:")
    for (i, spec) in enumerate(a_spec[1:10])
        println("  [$i]: ", spec)
    end
catch e
    println("Error getting A spec: ", e)
end
