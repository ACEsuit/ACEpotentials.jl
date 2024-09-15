
@testset "Vref" begin; include("test_Vref.jl"); end
@testset "Radial Envelopes" begin; include("test_radial_envelopes.jl"); end
@testset "Radial Transforms" begin; include("test_radial_transforms.jl"); end
@testset "Rnlrzz Basis" begin; include("test_Rnl.jl"); end
@testset "Pair Basis" begin; include("test_pair_basis.jl"); end
@testset "ACE Model" begin; include("test_ace.jl"); end 
@testset "ACE Calculator" begin; include("test_calculator.jl"); end
