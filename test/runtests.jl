using ACEpotentials, Test, LazyArtifacts

@testset "ACEpotentials.jl" begin

    # core package functionality 
    @testset "Models" begin include("models/test_models.jl") end 

    # fitting tests 
    @testset "Test silicon" begin include("test_silicon.jl") end
    @testset "Test recomputation of weights" begin include("test_recompw.jl") end

    # json interface tests 
    @testset "Test JSON interface" begin include("test_json.jl") end
    @testset "Test IO" begin include("test_io.jl")  end

    # make sure miscellaneous and weird bugs 
    @testset "Weird bugs" begin include("test_bugs.jl") end

    # new ET backend tests
    @testset "ET ACE" begin include("etmodels/test_etace.jl") end
    @testset "ET OneBody" begin include("etmodels/test_etonebody.jl") end
    @testset "ET Pair" begin include("etmodels/test_etpair.jl") end
    @testset "ET Calculators" begin include("et_models/test_et_calculators.jl") end

    # ACE1 compatibility tests 
    # TODO: these tests need to be revived either by creating a JSON 
    #       of test data, or by updating ACE1/ACE1x/JuLIP to be compatible. 
    # @testset "ACE1 Compat" begin include("ace1/test_ace1_compat.jl"); end 
end
