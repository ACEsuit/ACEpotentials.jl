# ---------------- Implementation of the repulsion restraint 

function _rep_dimer_data_atomsbase(
                  model::ACEPotential{<: ACEModel};
                  weight=0.01, 
                  energy_key=:energy,
                  group_key=:config_type,
                  kwargs...
            ) 
   B_pair = model.model.pairbasis
   zz = B_pair._i2z
   restraints = [] 

   for i = 1:length(zz), j = i:length(zz)
      z1, z2 = zz[i], zz[j]
      r0_est = 1.0   # could try to get this from the model meta-data 
      _rin = r0_est / 100  # can't take 0 since we'd end up with ∞ / ∞
      T_ij = model.model.pairbasis.transforms[i, j]
      env_ij = model.model.pairbasis.envelopes[i, j]
      env_rin = ACEpotentials.Models.evaluate(env_ij, _rin, T_ij(_rin))

      a1 = Atom(z1, zeros(3)u"Å")
      a2 = Atom(z2, [_rin, 0, 0]u"Å")
      cell = tuple([SA[_rin+1, 0.0, 0.0], SA[0.0, 1.0, 0.0], SA[0.0, 0.0, 1.0]]u"Å" ...)
      pbc = (false, false, false) 
      system = FlexibleSystem([a1, a2], cell, pbc)
      system.data[energy_key] = env_rin
      system.data[:config_type] = "restraint"
      
      data = AtomsData(system;
            energy_key = energy_key, 
            force_key  = nothing, 
            virial_key = nothing, 
            weights = Dict("restraint" => Dict("E" => weight)), 
            v_ref = _get_Vref(model)
            )
      
      push!(restraints, data)
   end

   return restraints
end

