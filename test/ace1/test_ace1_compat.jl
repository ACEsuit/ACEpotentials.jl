
# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

##

include(@__DIR__() * "/ace1_testutils.jl")

@info(
""" 
==================================
=== Testing ACE1 compatibility === 
==================================
""")


##
# [1] 
# a first test that was used to write the original 
# ACE1 compat module and tests 

params = ( elements = [:Si,], 
           order = 3, 
           transform = (:agnesi, 2, 2),
           totaldegree = [12, 10, 8], 
           pure = false, 
           pure2b = false,
           pair_envelope = (:r, 1),
           rcut = 5.5,
           Eref = [:Si => -1.234 ]
         )

ACE1_TestUtils.check_compat(params) 

##
# [2] 
# same as [1] but with auto cutoff, a different pair envelope, 
# different transforms, and a different chemical element, 
# different choice of degrees, different element   
# 

params = ( elements = [:C,], 
           order = 3, 
           transform = (:agnesi, 2, 4),
           totaldegree = [14, 12, 10], 
           pure = false, 
           pure2b = false,
           pair_transform = (:agnesi, 1, 3), 
           pair_envelope = (:r, 3),
           Eref = [:C => -1.234 ]
         )

ACE1_TestUtils.check_compat(params) 

##
# [3] 
# A minimal example with as many defaults as possible
# 

params = ( elements = [:W,], 
           order = 2, 
           totaldegree = 10, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 

## 
# [4] 
# First multi-species examples 

# NB : Ti, Al is a bad example because default bondlengths 
#      are the same. This can avoid some non-trivial behaviour 

params = ( elements = [:Al, :Cu,], 
           order = 3, 
           totaldegree = 8, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 

## [5] 
# second multi-species example with three elements 
# and a few small changes to the basis 

params = ( elements = [:Al, :Ti, :C], 
           order = 2, 
           totaldegree = 8, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 


## [6] 
# Confirm that the smoothness prior is the same in ACE1x and ACEpotentials 

import ACE1, ACE1x, JuLIP, Random 
using LinearAlgebra, ACEpotentials 
M = ACEpotentials.Models 
ACE1compat = ACEpotentials.ACE1compat
rng = Random.MersenneTwister(1234)

##

@info("Testing scaling of smoothness priors")

params = ( elements = [:Si,], 
           order = 3, 
           transform = (:agnesi, 2, 2),
           totaldegree = [16, 13, 11], 
           pure = false, 
           pure2b = false,
           pair_envelope = (:r, 1),
           rcut = 5.5,
           Eref = [:Si => -1.234 ]
         )

ACE1_TestUtils.check_compat(params)         

_p = (p = 1, wL = 1.5) 
ACE1_TestUtils.compare_smoothness_prior(params, :algebraic, _p, _p)

_p = (p = 2, wL = 1.5) 
ACE1_TestUtils.compare_smoothness_prior(params, :algebraic, _p, _p)

_p = (p = 4, wL = 1.5) 
ACE1_TestUtils.compare_smoothness_prior(params, :algebraic, _p, _p) 

ACE1_TestUtils.compare_smoothness_prior(
        params, :exponential, (al = 1.5, an = 1.0), (wl = 2/3, wn = 1.0) )

ACE1_TestUtils.compare_smoothness_prior(
        params, :exponential, (al = 0.234, an = 0.321), 
                              (wl = 1 / 0.234, wn = 1/0.321) )

ACE1_TestUtils.compare_smoothness_prior(
        params, :gaussian, (σl = 2.0, σn = 2.0), 
                           (wl = 1/sqrt(2), wn = 1/sqrt(2)) )

ACE1_TestUtils.compare_smoothness_prior(
        params, :gaussian, (σl = 1.234, σn = 0.89), 
                           (wl = 1/sqrt(1.234), wn = 1/sqrt(0.89)) )

