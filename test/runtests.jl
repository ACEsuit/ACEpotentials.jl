using ACEpotentials, Test, LazyArtifacts

@testset "ACEpotentials.jl" begin

    @testset "Test silicon" begin include("test_silicon.jl") end
    @testset "Test recomputation of weights" begin include("test_recompw.jl") end

    # outdated
    @testset "Read data" begin include("outdated/test_data.jl") end 
    @testset "Basis" begin include("outdated/test_basis.jl") end 
    @testset "Solver" begin include("outdated/test_solver.jl") end 
    @testset "Fit ACE" begin include("outdated/test_fit.jl") end 
    @testset "Read params" begin include("outdated/test_read_params.jl") end 
    #@testset "Test ace_fit.jl script" begin include("outdated/test_ace_fit.jl") end

end
