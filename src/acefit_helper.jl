function count_observations(filename, energy_key, force_key, virial_key)
    observations = 0
    for dict in ExtXYZ.iread_frames(filename)
        atoms = JuLIP._extxyz_dict_to_atoms(dict)
        for key in keys(atoms.data)
            if lowercase(key) == lowercase(energy_key)
                observations += 1
            elseif lowercase(key) == lowercase(force_key)
                observations += 3*length(atoms)
            elseif lowercase(key) == lowercase(virial_key)
                observations += 6
            end
        end
    end
    return observations
end

function _atoms_to_data(atoms, v_ref, weights, energy_key=nothing, force_key=nothing, virial_key=nothing)

    # wcw todo: subtract force and virial references as well

    energy = nothing  # wcw: these change type in the loop ... revise?
    forces = nothing
    virial = nothing
    config_type = "default"
    for key in keys(atoms.data)
        if lowercase(key)=="config_type"; config_type=atoms.data[key].data; end
    end
    for key in keys(atoms.data)
        if !isnothing(energy_key) && lowercase(key)==lowercase(energy_key)
            w = (config_type in keys(weights)) ? weights[config_type]["E"] : weights["default"]["E"]
            energy_ref = JuLIP.energy(v_ref, atoms)
            energy = atoms.data[key].data - energy_ref
            energy = ObsPotentialEnergy(energy, w, energy_ref)
        end
        if !isnothing(force_key) && lowercase(key)==lowercase(force_key)
            w = (config_type in keys(weights)) ? weights[config_type]["F"] : weights["default"]["F"]
            forces = ObsForces(atoms.data[key].data[:], w)
        end
        if !isnothing(virial_key) && lowercase(key)==lowercase(virial_key)
            w = (config_type in keys(weights)) ? weights[config_type]["V"] : weights["default"]["V"]
            m = SMatrix{3,3}(atoms.data[key].data)
            virial = ObsVirial(m, w)
        end
    end
    obs = Any[energy]
    if !isnothing(forces)
        push!(obs, forces)
    end
    if !isnothing(virial)
        insert!(obs, 1, virial)
    end
    return ACEfit.Dat(atoms, config_type, obs)
end

function error_llsq(data, approx, exact)

   errors = approx - exact
   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0, "F"=>0, "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   config_norms = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dat in data
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_norms, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
   end

   i = 1
   for dat in data
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         obs_values = exact[i:i+obs_len-1]
         # TODO: we store the ref energy because it is used for the relrmse
         #       calculation ... but does it make sense to use the total energy?
         if hasproperty(o, :E)
            obs_values = obs_values .+ o.E_ref
         end
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
            obs_values = obs_values ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         obs_norm = sum(obs_values.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_norms["set"]["E"] += obs_norm
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
            config_norms[dat.configtype]["E"] += obs_norm
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_norms["set"]["F"] += obs_norm
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
            config_norms[dat.configtype]["F"] += obs_norm
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_norms["set"]["V"] += obs_norm
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
            config_norms[dat.configtype]["V"] += obs_norm
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] = sqrt(config_errors[i][j] / config_counts[i][j])
         config_norms[i][j] = sqrt(config_norms[i][j] / config_counts[i][j])
         config_norms[i][j] = config_errors[i][j] / config_norms[i][j]
      end
   end

   return Dict("rmse"=>config_errors, "relrmse"=>config_norms)

end

function error_llsq_new(params, approx, exact)

   v_ref = OneBody(convert(Dict{String,Any},params["e0"]))
   energy_key = params["data"]["energy_key"]
   force_key = params["data"]["force_key"]
   virial_key = params["data"]["virial_key"]
   weights = params["weights"]

   errors = approx - exact

   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0, "F"=>0, "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   config_norms = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dict in ExtXYZ.iread_frames(params["data"]["fname"])
       atoms = JuLIP._extxyz_dict_to_atoms(dict)
       dat = _atoms_to_data(atoms, v_ref, weights, energy_key, force_key, virial_key)
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_norms, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
    end

   i = 1
   for dict in ExtXYZ.iread_frames(params["data"]["fname"])
      atoms = JuLIP._extxyz_dict_to_atoms(dict)
      dat = _atoms_to_data(atoms, v_ref, weights, energy_key, force_key, virial_key)
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         obs_values = exact[i:i+obs_len-1]
         # TODO: we store the ref energy because it is used for the relrmse
         #       calculation ... but does it make sense to use the total energy?
         if hasproperty(o, :E)
            obs_values = obs_values .+ o.E_ref
         end
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
            obs_values = obs_values ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         obs_norm = sum(obs_values.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_norms["set"]["E"] += obs_norm
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
            config_norms[dat.configtype]["E"] += obs_norm
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_norms["set"]["F"] += obs_norm
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
            config_norms[dat.configtype]["F"] += obs_norm
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_norms["set"]["V"] += obs_norm
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
            config_norms[dat.configtype]["V"] += obs_norm
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] = sqrt(config_errors[i][j] / config_counts[i][j])
         config_norms[i][j] = sqrt(config_norms[i][j] / config_counts[i][j])
         config_norms[i][j] = config_errors[i][j] / config_norms[i][j]
      end
   end

   return Dict("rmse"=>config_errors, "relrmse"=>config_norms)

end
