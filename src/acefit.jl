import ACEfit
import JuLIP: Atoms, energy, forces, mat
using StaticArrays: SVector

struct AtomsData <: ACEfit.AbstractData
    atoms::Atoms
    energy_key
    force_key
    virial_key
    weights
    vref
    function AtomsData(atoms::Atoms,
                       energy_key, force_key, virial_key,
                       weights, vref)

        # set energy, force, and virial keys for this configuration
        # ("nothing" indicates data that are absent or ignored)
        ek, fk, vk = nothing, nothing, nothing
        for key in keys(atoms.data)
            if lowercase(key) == lowercase(energy_key)
                ek = key
            elseif lowercase(key) == lowercase(force_key)
                fk = key
            elseif lowercase(key) == lowercase(virial_key)
                vk = key
            end
        end

        # set weights for this configuration
        if "default" in keys(weights)
            w = weights["default"]
        else
            w = Dict("E"=>1.0, "F"=>1.0, "V"=>1.0)
        end
        for (key, val) in atoms.data
            if lowercase(key)=="config_type" && (lowercase(val.data) in lowercase.(keys(weights)))
                w = weights[val.data]
            end
        end

        return new(atoms, ek, fk, vk, w, vref)
    end
end

function ACEfit.count_observations(d::AtomsData)
    return !isnothing(d.energy_key) +
           3*length(d.atoms)*!isnothing(d.force_key) +
           6*!isnothing(d.virial_key)
end

function ACEfit.feature_matrix(d::AtomsData, basis)
    dm = Array{Float64}(undef, ACEfit.count_observations(d), length(basis))
    i = 1
    if !isnothing(d.energy_key)
        dm[i,:] .= energy(basis, d.atoms)
        i += 1
    end
    if !isnothing(d.force_key)
        f = mat.(forces(basis, d.atoms))
        for j in 1:length(basis)
            dm[i:i+3*length(d.atoms)-1,j] .= f[j][:]
        end
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        v = virial(basis, d.atoms)
        for j in 1:length(basis)
            dm[i:i+5,j] .= v[j][SVector(1,5,9,6,3,2)]
        end
        i += 5
    end
    return dm
end

function ACEfit.target_vector(d::AtomsData)
    y = Array{Float64}(undef, ACEfit.count_observations(d))
    i = 1
    if !isnothing(d.energy_key)
        e = d.atoms.data[d.energy_key].data
        y[i] = e - energy(d.vref, d.atoms)
        i += 1
    end
    if !isnothing(d.force_key)
        f = mat(d.atoms.data[d.force_key].data)
        y[i:i+3*length(d.atoms)-1] .= f[:]
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        v = vec(d.atoms.data[d.virial_key].data)
        y[i:i+5] .= v[SVector(1,5,9,6,3,2)]
        i += 5
    end
    return y
end

function ACEfit.weight_vector(d::AtomsData)
    w = Array{Float64}(undef, ACEfit.count_observations(d))
    i = 1
    if !isnothing(d.energy_key)
        w[i] = d.weights["E"] / sqrt(length(d.atoms))
        i += 1
    end
    if !isnothing(d.force_key)
        w[i:i+3*length(d.atoms)-1] .= d.weights["F"]
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        w[i:i+5] .= d.weights["V"] / sqrt(length(d.atoms))
        i += 5
    end
    return w
end

function config_type(d::AtomsData)
    config_type = missing
    for (k,v) in d.atoms.data
        if (lowercase(k)=="config_type")
            config_type = v.data
        end
    end
    return config_type
end

function llsq_errors(data, model)

   mae = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
   rmse = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
   num = Dict("E"=>0, "F"=>0, "V"=>0)

   config_types = []
   config_mae = Dict{String,Any}()
   config_rmse = Dict{String,Any}()
   config_num = Dict{String,Any}()

   for d in data

       ct = config_type(d)
       if !(ct in config_types)
          push!(config_types, ct)
          merge!(config_rmse, Dict(ct=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_mae, Dict(ct=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_num, Dict(ct=>Dict("E"=>0, "F"=>0, "V"=>0)))
       end

       # energy errors
       if !isnothing(d.energy_key)
           estim = energy(model, d.atoms) / length(d.atoms)
           exact = d.atoms.data[d.energy_key].data / length(d.atoms)
           mae["E"] += abs(estim-exact)
           rmse["E"] += (estim-exact)^2
           num["E"] += 1
           config_mae[ct]["E"] += abs(estim-exact)
           config_rmse[ct]["E"] += (estim-exact)^2
           config_num[ct]["E"] += 1
       end

       # force errors
       if !isnothing(d.force_key)
           estim = mat(forces(model, d.atoms))
           exact = mat(d.atoms.data[d.force_key].data)
           mae["F"] += sum(abs.(estim-exact))
           rmse["F"] += sum((estim-exact).^2)
           num["F"] += 3*length(d.atoms)
           config_mae[ct]["F"] += sum(abs.(estim-exact))
           config_rmse[ct]["F"] += sum((estim-exact).^2)
           config_num[ct]["F"] += 3*length(d.atoms)
       end

       # virial errors
       if !isnothing(d.virial_key)
           estim = virial(model, d.atoms)[SVector(1,5,9,6,3,2)] ./ length(d.atoms)
           exact = d.atoms.data[d.virial_key].data[SVector(1,5,9,6,3,2)] ./ length(d.atoms)
           mae["V"] += sum(abs.(estim-exact))
           rmse["V"] += sum((estim-exact).^2)
           num["V"] += 6
           config_mae[ct]["V"] += sum(abs.(estim-exact))
           config_rmse[ct]["V"] += sum((estim-exact).^2)
           config_num[ct]["V"] += 6
       end
    end

    # finalize errors
    for (k,n) in num
        (n==0) && continue
        rmse[k] = sqrt(rmse[k] / n)
        mae[k] = mae[k] / n
    end
    errors = Dict("mae"=>mae, "rmse"=>rmse)

    # finalize config errors
    for ct in config_types
        for (k,cn) in config_num[ct]
            (cn==0) && continue
            config_rmse[ct][k] = sqrt(config_rmse[ct][k] / cn)
            config_mae[ct][k] = config_mae[ct][k] / cn
        end
    end
    config_errors = Dict("mae"=>config_mae, "rmse"=>config_rmse)

    # merge errors into config_errors and return
    merge!(config_errors["mae"], Dict("set"=>mae))
    merge!(config_errors["rmse"], Dict("set"=>rmse))
    return config_errors
end
