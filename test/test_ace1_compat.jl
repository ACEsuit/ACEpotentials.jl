
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

##

using Random, Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

##

