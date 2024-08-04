# This script is to explore the differences between the ACE1 models and the new 
# models. This is to help bring the two to feature parity so that ACE1 
# can be retired. 

using Random
using ACEpotentials, Lux
M = ACEpotentials.Models

function matching_bases(; Z = :Si, order = 3, totaldegree = 10, 
                          rcut = 5.5)

   elements = [Z, ]                          
   model1 = acemodel(elements = elements, 
                     order = order, 
                     transform = (:agnesi, 2, 2),
                     totaldegree = totaldegree, 
                     pure = false, 
                     pure2b = false, 
                     pair_envelope = (:r, 1), 
                     rcut = rcut,  )

   rin0cuts = M._default_rin0cuts(elements) #; rcutfactor = 2.29167)
   rin0cuts = SMatrix{1,1}((;rin0cuts[1]..., :rcut => 5.5))

   model2 = M.ace_model(; elements = elements, 
                       order = order,               # correlation order 
                       Ytype = :spherical,              # solid vs spherical harmonics
                       level = M.TotalDegree(),     # how to calculate the weights to give to a basis function
                       max_level = totaldegree,     # maximum level of the basis functions
                       pair_maxn = totaldegree,     # maximum number of basis functions for the pair potential 
                       init_WB = :zeros,            # how to initialize the ACE basis parmeters
                       init_Wpair = :onehot,         # how to initialize the pair potential parameters
                       init_Wradial = :onehot, 
                       pair_transform = (:agnesi, 1, 3), 
                       pair_learnable = true, 
                       rin0cuts = rin0cuts, 
                     )

   ps, st = Lux.setup(Random.GLOBAL_RNG, model2)
   ps_r = ps.rbasis
   st_r = st.rbasis

   # extract the radial basis 
   rbasis1 = model1.basis.BB[2].pibasis.basis1p.J
   rbasis2 = model2.rbasis
   k = length(rbasis1.J.A)

   # transform old coefficients to new coefficients to make them match 
   rbasis1.J.A[:] .= rbasis2.polys.A[1:k]
   rbasis1.J.B[:] .= rbasis2.polys.B[1:k]
   rbasis1.J.C[:] .= rbasis2.polys.C[1:k]
   rbasis1.J.A[2] /= rbasis1.J.A[1] 
   rbasis1.J.B[2] /= rbasis1.J.A[1]

   # fix the basis1 spec 
   _spec1 = ACE1.get_nl(model1.basis.BB[2])
   spec1 = [ [ (n = b.n, l = b.l) for b in bb ] for bb in _spec1 ]
   spec2 = M.get_nnll_spec(model2.tensor)
   spec1 = sort.(spec1)
   spec2 = sort.(spec2)
   Nb = length(spec2)
   idx2in1 = [ findfirst( Ref(bb) .== spec1 ) for bb in spec2 ]
   @show length(idx2in1) == Nb

   idx_del = setdiff((1:size(model1.basis.BB[2].A2Bmaps[1], 1)), idx2in1)
   model1.basis.BB[2].A2Bmaps[1][idx_del, :] .= 0 
   BB2 = ACE1.RPI.remove_zeros(ACE1._cleanup(model1.basis.BB[2]))
   model1.basis.BB[2] = BB2
   
   # wrap the model into a calculator, which turns it into a potential...
   calc_model2 = M.ACEPotential(model2)

   return model1, model2, calc_model2
end




