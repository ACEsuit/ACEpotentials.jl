
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
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
# different choice of degrees  
# 

params = ( elements = [:Si,], 
           order = 3, 
           transform = (:agnesi, 2, 4),
           totaldegree = [14, 12, 10], 
           pure = false, 
           pure2b = false,
           pair_transform = (:agnesi, 1, 3), 
           pair_envelope = (:r, 3),
           Eref = [:Si => -1.234 ]
         )

ACE1_TestUtils.check_compat(params) 

##
# [3] 
# A minimal example with as many defaults as possible 
# 

params = ( elements = [:Si,], 
           order = 2, 
           totaldegree = 10, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 

