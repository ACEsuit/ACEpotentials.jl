using ACEpotentials, Test, LazyArtifacts

##

@testset "ACEpotentials.jl" begin

    @testset "Read data" begin include("test_data.jl") end 

    @testset "Basis" begin include("test_basis.jl") end 

    @testset "Solver" begin include("test_solver.jl") end 

    @testset "Fit ACE" begin include("test_fit.jl") end 

    @testset "Read params" begin include("test_read_params.jl") end 

    @testset "Test silicon" begin include("test_silicon.jl") end

    @testset "Test recomputation of weights" begin include("test_recompw.jl") end

#    @testset "Test ace_fit.jl script" begin include("test_ace_fit.jl") end

end
