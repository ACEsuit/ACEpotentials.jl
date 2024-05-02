
module Experimental 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     experimental new ACE kernels 
#     largely untested, use with care

import UltraFastACE 
using ACE1x: ACE1Model 
using JuLIP.MLIPs: SumIP 

fast_evaluator(model::ACE1Model; kwargs...) = 
      fast_evaluator(model.potential; kwargs...)

function fast_evaluator(pot::SumIP; n_spl_points = 10_000)
   uf_ace = UltraFastACE.uface_from_ace1(pot; n_spl_points = n_spl_points)
   return UltraFastACE.UFACE_JuLIP(uf_ace)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


end