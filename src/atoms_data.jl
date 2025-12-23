import ACEfit, AtomsBase, AtomsCalculators
using PrettyTables
using OrderedCollections
using StaticArrays: SVector

import AtomsBase: AbstractSystem, FlexibleSystem, atomkeys
import AtomsCalculators: energy_forces_virial, potential_energy, forces, virial 
import ACEpotentials.Models: energy_forces_virial_basis, length_basis

import ExtXYZ 

# Some utilities 


function _f_vec(f::AbstractVector{SVector{D, T}}) where {D, T}
    return reinterpret(T, f)
end

function _f_vec(f::AbstractVector{Vector{T}}) where {T}
    return reduce(vcat, f)
end

function _f_mat(f::AbstractMatrix{SVector{D, T}}) where {D, T} 
    return reinterpret(T, f)
end

function _getfuzzy(coll, key)
    k = _find_similar_key(coll, key) 
    if k != nothing 
        return coll[k] 
    end 
    error("Couldn't find $key or similar in collection with keys $(keys(coll))")
end

_issimilarkey(k1, k2) = lowercase(String(k1)) == lowercase(String(k2))
_issimilarkey(k1::Nothing, k2) = false 
_issimilarkey(k1, k2::Nothing) = false
_issimilarkey(k1::Nothing, k2::Nothing) = false

function _find_similar_key(coll, key) 
    for k in keys(coll) 
        if _issimilarkey(k, key)
            return k 
        end
    end
    return nothing 
end

function _find_similar_key(sys::AbstractSystem, key) 
    for k in keys(sys)
        if _issimilarkey(k, key)
            return k 
        end
    end
    for k in atomkeys(sys)
        if _issimilarkey(k, key)
            return k 
        end
    end
    return nothing 
end

function _get_data_fuzzy(sys::AbstractSystem, key)
    k = _find_similar_key(sys, key) 
    if k == nothing 
        error("Couldn't find $key or similar in collection with keys $(keys(sys))")
    end 
    if haskey(sys, k)
        return sys[k]
    end
    return sys[:, k]
end

_has_similar_key(coll, key) = (_find_similar_key(coll, key) != nothing)

function _get_data(sys::AbstractSystem, key)
    if haskey(sys, key)
        return sys[key]
    elseif hasatomkey(sys, key)
        return sys[:, key]
    else
        error("Couldn't find $key in System")
    end
end


export AtomsData, 
       print_rmse_table, print_mae_table, print_errors_tables

struct AtomsData <: ACEfit.AbstractData
    system  # an AtomsBase.AbstractSystem object 
    energy_key::Union{String, Symbol, Nothing}
    force_key::Union{String, Symbol, Nothing}
    virial_key::Union{String, Symbol, Nothing}
    weights
    energy_ref
    function AtomsData(system;
                       energy_key=nothing,
                       force_key=nothing,
                       virial_key=nothing,
                       weights=Dict("default"=>Dict("E"=>1.0, "F"=>1.0, "V"=>1.0)),
                       v_ref=nothing, 
                       weight_key="config_type")

        # set energy, force, and virial keys for this configuration
        # ("nothing" indicates data that are absent or ignored)
        ek = _find_similar_key(system, energy_key)
        fk = _find_similar_key(system, force_key)
        vk = _find_similar_key(system, virial_key)

        # set weights for this configuration
        w = nothing
        if _has_similar_key(system, weight_key)
            cfg_type = _get_data_fuzzy(system, weight_key)
            if _has_similar_key(weights, cfg_type)
                w = _getfuzzy(weights, cfg_type)
            end
        end 
        if isnothing(w) 
            if "default" in keys(weights)
                w = weights["default"]
            else
                w = Dict("E"=>1.0, "F"=>1.0, "V"=>1.0)
            end
        end

        if isnothing(v_ref)
            e_ref = 0.0
        else
            e_ref = potential_energy(system, v_ref)
        end

        return new(system, ek, fk, vk, w, e_ref)
    end
end

function ACEfit.count_observations(d::AtomsData)
    return !isnothing(d.energy_key) +
           3*length(d.system)*!isnothing(d.force_key) +
           6*!isnothing(d.virial_key)
end


# TODO: the usage of ustrip in feature_matrix is a bit dangerous. 
#       it needs to be revisited, basically to unsure that all dimensions 
#       are in the same units.

function ACEfit.feature_matrix(d::AtomsData, model; kwargs...)
    dm = Array{Float64}(undef, ACEfit.count_observations(d), length_basis(model))
    efv = energy_forces_virial_basis(d.system, model)
    i = 1
    if !isnothing(d.energy_key)
        dm[i,:] .= ustrip(efv.energy)
        i += 1
    end
    if !isnothing(d.force_key)
        dm[i:i+3*length(d.system)-1, :] .= ustrip.(_f_mat(efv.forces))
        i += 3*length(d.system)
    end
    if !isnothing(d.virial_key)
        for j in 1:length_basis(model)
            dm[i:i+5,j] .= ustrip.( (efv.virial[j])[SVector(1,5,9,6,3,2)] )
        end
        i += 6
    end
    return dm
end



function ACEfit.target_vector(d::AtomsData; kwargs...)
    y = Array{Float64}(undef, ACEfit.count_observations(d))
    i = 1
    if !isnothing(d.energy_key)
        e = _get_data(d.system, d.energy_key) 
        y[i] = e - d.energy_ref
        i += 1
    end
    if !isnothing(d.force_key)
        # f = mat(d.system.data[d.force_key].data)
        f = _f_vec( _get_data(d.system, d.force_key) )
        y[i:i+3*length(d.system)-1] .= f
        i += 3*length(d.system)
    end
    if !isnothing(d.virial_key)
        try 
        # the following hack is necessary for 3-atom cells:
        #   https://github.com/JuliaMolSim/JuLIP.jl/issues/166
        #v = vec(d.system.data[d.virial_key].data)
        v = vec(hcat(_get_data(d.system, d.virial_key)...))
        y[i:i+5] .= v[SVector(1,5,9,6,3,2)]
        i += 6
        catch 
            @show keys(d.system.data)
            @show length(d.system)
            @show d.virial_key 
        end
    end
    return y
end

function ACEfit.weight_vector(d::AtomsData; kwargs...)
    w = Array{Float64}(undef, ACEfit.count_observations(d))
    i = 1
    if !isnothing(d.energy_key)
        w[i] = d.weights["E"] / sqrt(length(d.system))
        i += 1
    end
    if !isnothing(d.force_key)
        w[i:i+3*length(d.system)-1] .= d.weights["F"]
        i += 3*length(d.system)
    end
    if !isnothing(d.virial_key)
        w[i:i+5] .= d.weights["V"] / sqrt(length(d.system))
        i += 6
    end
    return w
end

function recompute_weights(raw_data;
                           energy_key=nothing, force_key=nothing, virial_key=nothing,
                           weights=Dict("default"=>Dict("E"=>1.0, "F"=>1.0, "V"=>1.0)))
    data = [ AtomsData(at; energy_key = energy_key, force_key=force_key,
                   virial_key = virial_key, weights = weights) for at in raw_data ]
    return ACEfit.assemble_weights(data)
end

function group_type(d::AtomsData; group_key="config_type")
    gk = _find_similar_key(d.system, group_key)
    if gk == nothing
        gt = "nil"
    else 
        gt = _get_data(d.system, gk)
    end

    return gt
end


function compute_errors(data::AbstractArray{AtomsData}, model; 
                       group_key="config_type", verbose=true,
                       return_efv = false
                       )

   mae = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
   rmse = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
   num = Dict("E"=>0, "F"=>0, "V"=>0)

   config_types = []
   config_mae = Dict{String,Any}()
   config_rmse = Dict{String,Any}()
   config_num = Dict{String,Any}()

   evf_dict = Dict("Epred" => [], "Eref" => [],
                   "Fpred" => [], "Fref" => [],
                   "Vpred" => [], "Vref" => [],
                   )

   for d in data

       c_t = group_type(d; group_key)
       if !(c_t in config_types)
          push!(config_types, c_t)
          merge!(config_rmse, Dict(c_t=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_mae, Dict(c_t=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_num, Dict(c_t=>Dict("E"=>0, "F"=>0, "V"=>0)))
       end

       efv = energy_forces_virial(d.system, model)

       # energy errors
       if !isnothing(d.energy_key)
           estim = ustrip(efv.energy) / length(d.system)
           exact = _get_data(d.system, d.energy_key) / length(d.system)
           mae["E"] += abs(estim-exact)
           rmse["E"] += (estim-exact)^2
           num["E"] += 1
           config_mae[c_t]["E"] += abs(estim-exact)
           config_rmse[c_t]["E"] += (estim-exact)^2
           config_num[c_t]["E"] += 1
           push!(evf_dict["Epred"], estim)
           push!(evf_dict["Eref"], exact)
       end

       # force errors
       if !isnothing(d.force_key)
           estim = ustrip.(_f_vec(efv.forces))
           exact = _f_vec(_get_data(d.system, d.force_key))
           mae["F"] += sum(abs.(estim-exact))
           rmse["F"] += sum((estim-exact).^2)
           num["F"] += 3*length(d.system)
           config_mae[c_t]["F"] += sum(abs, estim - exact)
           config_rmse[c_t]["F"] += sum(abs2, estim - exact)
           config_num[c_t]["F"] += 3*length(d.system)
           push!(evf_dict["Fpred"], estim)
           push!(evf_dict["Fref"], exact)
       end

       # virial errors
       if !isnothing(d.virial_key)
           estim = ustrip.( efv.virial[SVector(1,5,9,6,3,2)] / length(d.system) )
           # the following hack is necessary for 3-atom cells:
           #   https://github.com/JuliaMolSim/JuLIP.jl/issues/166
           #exact = d.system.data[d.virial_key].data[SVector(1,5,9,6,3,2)] ./ length(d.system)
        #    exact = hcat(d.system.data[d.virial_key].data...)[SVector(1,5,9,6,3,2)] ./ length(d.system)
           v = vec(hcat(_get_data(d.system, d.virial_key)...))
           exact = v[SVector(1,5,9,6,3,2)] / length(d.system)
           mae["V"] += sum(abs.(estim-exact))
           rmse["V"] += sum((estim-exact).^2)
           num["V"] += 6
           config_mae[c_t]["V"] += sum(abs, estim-exact)
           config_rmse[c_t]["V"] += sum(abs2, estim-exact)
           config_num[c_t]["V"] += 6
           push!(evf_dict["Vpred"], estim)
           push!(evf_dict["Vref"], exact)
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
    for c_t in config_types
        for (k,c_n) in config_num[c_t]
            (c_n==0) && continue
            config_rmse[c_t][k] = sqrt(config_rmse[c_t][k] / c_n)
            config_mae[c_t][k] = config_mae[c_t][k] / c_n
        end
    end
    config_errors = Dict("mae"=>config_mae, "rmse"=>config_rmse)

    # merge errors into config_errors and return
    push!(config_types, "set")
    merge!(config_errors["mae"], Dict("set"=>mae))
    merge!(config_errors["rmse"], Dict("set"=>rmse))

    if verbose
        print_errors_tables(config_errors)
    end 

    if return_efv
        return config_errors, evf_dict
    else
        return config_errors
    end
end


function print_errors_tables(config_errors::AbstractDict)
    print_rmse_table(config_errors)
    print_mae_table(config_errors)
end

function _print_err_tbl(D::AbstractDict)
    header = ["Type", "E [meV]", "F [eV/A]", "V [meV]"]
    config_types = setdiff(collect(keys(D)), ["set",])
    push!(config_types, "set")
    table = hcat(
        config_types,
        [1000*D[c_t]["E"] for c_t in config_types],
        [D[c_t]["F"] for c_t in config_types],
        [1000*D[c_t]["V"] for c_t in config_types],
    )
    pretty_table(
        table; header=header,
        body_hlines=[length(config_types)-1],
        formatters=ft_printf("%5.3f"),
        crop = :horizontal)

end

function print_rmse_table(config_errors::AbstractDict; header=true)
    if header; (@info "RMSE Table"); end 
    _print_err_tbl(config_errors["rmse"])
end

function print_mae_table(config_errors::AbstractDict; header=true)
    if header; (@info "MAE Table"); end 
    _print_err_tbl(config_errors["mae"])
end


function assess_dataset(data; kwargs...)
    config_types = []

    n_configs = OrderedDict{String,Integer}()
    n_environments = OrderedDict{String,Integer}()
    n_energies = OrderedDict{String,Integer}()
    n_forces = OrderedDict{String,Integer}()
    n_virials = OrderedDict{String,Integer}()

    for d in data
        if haskey(kwargs, :group_key)
            c_t = group_type(d; group_key=kwargs[:group_key])
        else
            c_t = group_type(d)
        end
        if !(c_t in config_types)
            push!(config_types, c_t)
            n_configs[c_t] = 0
            n_environments[c_t] = 0
            n_energies[c_t] = 0
            n_forces[c_t] = 0
            n_virials[c_t] = 0
        end
        n_configs[c_t] += 1
        n_environments[c_t] += length(d)
        _has_energy(d; kwargs...) && (n_energies[c_t] += 1)
        _has_forces(d; kwargs...) && (n_forces[c_t] += 3*length(d))
        _has_virial(d; kwargs...) && (n_virials[c_t] += 6)
    end

    n_configs = collect(values(n_configs))
    n_environments = collect(values(n_environments))
    n_energies = collect(values(n_energies))
    n_forces = collect(values(n_forces))
    n_virials = collect(values(n_virials))

    header = ["Type", "#Configs", "#Envs", "#E", "#F", "#V"]
    table = hcat(
        config_types, n_configs, n_environments,
        n_energies, n_forces, n_virials)
    tot = [
        "total", sum(n_configs), sum(n_environments),
         sum(n_energies), sum(n_forces), sum(n_virials)]
    miss = [
        "missing", 0, 0,
        tot[4]-tot[2], 3*tot[3]-tot[5], 6*tot[2]-tot[6]]
    table = vcat(table, permutedims(tot), permutedims(miss))
    pretty_table(table; header=header, body_hlines=[length(n_configs)], crop = :horizontal)

end

_has_energy(data::AtomsData; kwargs...) = !isnothing(data.energy_key)
_has_forces(data::AtomsData; kwargs...) = !isnothing(data.force_key)
_has_virial(data::AtomsData; kwargs...) = !isnothing(data.virial_key)


Base.length(a::AtomsData) = length(a.system)