using ACE1pack, Test, LazyArtifacts

##

@testset "ACE1pack.jl" begin

    @testset "Read data" begin include("test_data.jl") end 

    @testset "Basis" begin include("test_basis.jl") end 

    @testset "Solver" begin include("test_solver.jl") end 

    @testset "Fit ACE" begin include("test_fit.jl") end 

    @testset "Test Silicon" begin include("test_silicon.jl") end 

    @testset "Read params" begin include("test_read_params.jl") end 

    @testset "Test silicon" begin include("test_silicon.jl") end

end
