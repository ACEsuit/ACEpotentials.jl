# ---------------- Implementation of the repulsion restraint 

function _rep_dimer_data_atomsbase(
   model;
   weight=0.01, 
   energy_key=:energy,
   group_key=:config_type,
   kwargs...
) 
   zz = model.basis.BB[1].zlist.list
   restraints = [] 
   B_pair = model.basis.BB[1] 
   if !isa(B_pair, ACE1.PolyPairBasis)
      error("repulsion restraints only implemented for PolyPairBasis")
   end

   for i = 1:length(zz), j = i:length(zz)
      z1, z2 = zz[i], zz[j]
      s1, s2 = chemical_symbol.((z1, z2))
      r0_est = 1.0   # could try to get this from the model meta-data 
      _rin = r0_est / 100  # can't take 0 since we'd end up with ∞ / ∞
      Pr_ij = B_pair.J[i, j]
      if !isa(Pr_ij, ACE1.OrthPolys.TransformedPolys)
         error("repulsion restraints only implemented for TransformedPolys")
      end
      envfun = Pr_ij.envelope 
      if !isa(envfun, ACE1.OrthPolys.PolyEnvelope)
         error("repulsion restraints only implemented for PolyEnvelope")
      end
      if !(envfun.p >= 0)
         error("repulsion restraints only implemented for PolyEnvelope with p >= 0")
      end
      env_rin = ACE1.evaluate(envfun, _rin)

      a1 = Atom(zz[1].z, zeros(3)u"Å")
      a2 = Atom(zz[2].z, [_rin, 0, 0]u"Å")
      cell = [ [_rin+1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]u"Å"
      boundary_conditions = [DirichletZero(), DirichletZero(), DirichletZero()]
      data = FlexibleSystem([a1, a2], cell, boundary_conditions)
      
      # add weight to the structure
      kwargs =[
         energy_key => env_rin, 
         group_key  => "restraint", 
         :energy_weight => weight,
      ]
      data = FlexibleSystem(data; kwargs...)

      push!(restraints, data)
   end

   return restraints
end

function _rep_dimer_data(model; 
                         weight = 0.01
                         )
   zz = model.basis.BB[1].zlist.list
   restraints = [] 
   restraint_weights = Dict("restraint" => Dict("E" => weight, "F" => 0.0, "V" => 0.0))
   B_pair = model.basis.BB[1] 
   if !isa(B_pair, ACE1.PolyPairBasis)
      error("repulsion restraints only implemented for PolyPairBasis")
   end

   for i = 1:length(zz), j = i:length(zz)
      z1, z2 = zz[i], zz[j]
      s1, s2 = chemical_symbol.((z1, z2))
      r0_est = 1.0   # could try to get this from the model meta-data 
      _rin = r0_est / 100  # can't take 0 since we'd end up with ∞ / ∞
      Pr_ij = B_pair.J[i, j]
      if !isa(Pr_ij, ACE1.OrthPolys.TransformedPolys)
         error("repulsion restraints only implemented for TransformedPolys")
      end
      envfun = Pr_ij.envelope 
      if !isa(envfun, ACE1.OrthPolys.PolyEnvelope)
         error("repulsion restraints only implemented for PolyEnvelope")
      end
      if !(envfun.p >= 0)
         error("repulsion restraints only implemented for PolyEnvelope with p >= 0")
      end
      env_rin = ACE1.evaluate(envfun, _rin)
      at = at_dimer(_rin, z1, z2)
      set_data!(at, "REF_energy", env_rin)
      set_data!(at, "config_type", "restraint")
      #  AtomsData(atoms::Atoms; energy_key, force_key, virial_key, weights, v_ref, weight_key)
      dat = ACEpotentials.AtomsData(at, energy_key = "REF_energy", 
                                    force_key = "REF_forces", 
                                    virial_key = "REF_virial", 
                                    weights = restraint_weights, 
                                    v_ref = model.Vref)
      push!(restraints, dat) 
   end
   
   return restraints
end
