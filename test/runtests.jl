using ACE1pack
using Test

@testset "ACE1pack.jl" begin

    include("test_data.jl")

    include("test_basis.jl")

    include("test_solver.jl")

    include("test_fit.jl")

    include("test_read_params.jl")

end
