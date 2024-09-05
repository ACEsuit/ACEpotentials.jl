using ACEpotentials, Test, LazyArtifacts

@testset "ACEpotentials.jl" begin

    @testset "Models" begin include("models/test_models.jl") end 

    # fitting tests 
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
    # TODO: these tests need to be revived either by creating a JSON 
    #       of test data, or by updating ACE1/ACE1x/JuLIP to be compatible. 
    # @testset "ACE1 Compat" begin include("ace1/test_ace1_compat.jl"); end 
end
