
function group_type(data; group_key=:config_type)
    key = Symbol(group_key)
    return haskey(data, key) ? data[key] : "default"
end

_has_energy(data; energy_key=:energy, kwargs...) = haskey(data, Symbol(energy_key))
_has_forces(data; force_key=:force, kwargs...)   = hasatomkey(data, Symbol(force_key))
_has_virial(data; virial_key=:virial, kwargs...) = haskey(data, Symbol(virial_key))


function linear_errors(data, model::ACE1x.ACE1Model; kwargs...)
    return linear_errors(data, ACEmd.ACEpotential(model.potential.components); kwargs...)
end


function linear_errors(
    data, 
    model::ACEmd.ACEpotential; 
    group_key="config_type", 
    verbose=true,
    energy_key = :energy,
    force_key  = :force,
    virial_key = :virial,
    kwargs... 
)

    mae = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
    rmse = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
    num = Dict("E"=>0, "F"=>0, "V"=>0)
 
    config_types = []
    config_mae = Dict{String,Any}()
    config_rmse = Dict{String,Any}()
    config_num = Dict{String,Any}()
 
    for d in data
 
        c_t = group_type(d; group_key)
        if !(c_t in config_types)
           push!(config_types, c_t)
           merge!(config_rmse, Dict(c_t=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
           merge!(config_mae, Dict(c_t=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
           merge!(config_num, Dict(c_t=>Dict("E"=>0, "F"=>0, "V"=>0)))
        end
 
        # energy errors
        if _has_energy(d; energy_key=energy_key)
            estim = ace_energy(model, d; energy_unit=NoUnits)
            exact = ustrip( d[energy_key] ) 
            Δ = (estim-exact) / length(d)
            config_mae[c_t]["E"] += abs(Δ)
            config_rmse[c_t]["E"] += abs2(Δ)
            config_num[c_t]["E"] += 1
        end
 
        # force errors
        if _has_forces(d; force_key=force_key)
            estim = ace_forces(model,d; energy_unit=NoUnits, length_unit=NoUnits) 
            exact = map( x->x[force_key], d)
            Δ = estim-exact
            config_mae[c_t]["F"] += sum(x->sum(abs,x), Δ)
            config_rmse[c_t]["F"] += sum(x->sum(abs2,x), Δ)
            config_num[c_t]["F"] += 3*length(d)
        end
 
        # virial errors
        if _has_virial(d; virial_key=virial_key)
            estim = ace_virial(model, d; energy_unit=NoUnits, length_unit=NoUnits)
            exact = ustrip.( d[virial_key] )
            Δ = (estim - exact)[SVector(1,5,9,6,3,2)] ./ length(d)
            config_mae[c_t]["V"] += sum(abs, Δ)
            config_rmse[c_t]["V"] += sum(abs2, Δ)
            config_num[c_t]["V"] += 6
       end
    end
    
    mae["E"]  = sum( v->v["E"],  values(config_mae)  )
    rmse["E"] = sum( v->v["E"],  values(config_rmse) )
    num["E"]  = sum( v->v["E"],  values(config_num)  ) 
    
    mae["F"]  = sum( v->v["F"],  values(config_mae)  )
    rmse["F"] = sum( v->v["F"],  values(config_rmse) )
    num["F"]  = sum( v->v["F"],  values(config_num)  ) 

    mae["V"]  = sum( v->v["V"],  values(config_mae)  )
    rmse["V"] = sum( v->v["V"],  values(config_rmse) )
    num["V"]  = sum( v->v["V"],  values(config_num)  ) 

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
 
    return config_errors, errors
end