using ACEpotentials, Test, LazyArtifacts

@testset "ACEpotentials.jl" begin

    @testset "Models" begin include("models/test_models.jl") end 


    @testset "Test silicon" begin include("test_silicon.jl") end
    @testset "Test recomputation of weights" begin include("test_recompw.jl") end

    # TODO: bring FIO back cf Issue #217
    # @testset "Test IO" begin include("test_io.jl")  end

    # experimental 
    # TODO move UF_ACE into ACEpotential properly
    # @testset "UF_ACE" begin include("test_uface.jl") end

    # weird stuff 
    @testset "Weird bugs" begin include("test_bugs.jl") end

    # ACE1 compatibility tests 
    @testset "ACE1 Compat" begin include("ace1/test_ace1_compat.jl"); end 

    # outdated
    # @testset "Read data" begin include("outdated/test_data.jl") end 
    # @testset "Basis" begin include("outdated/test_basis.jl") end 
    # @testset "Solver" begin include("outdated/test_solver.jl") end 
    # @testset "Fit ACE" begin include("outdated/test_fit.jl") end 
    # @testset "Read params" begin include("outdated/test_read_params.jl") end 
    # @testset "Test ace_fit.jl script" begin include("outdated/test_ace_fit.jl") end
end
