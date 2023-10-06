
function group_type(data; group_key=:config_type)
    key = Symbol(group_key)
    return haskey(data, key) ? data[key] : "default"
end

_has_energy(data; energy_key=:energy, kwargs...) = haskey(data, Symbol(energy_key))
_has_forces(data; force_key=:force, kwargs...)   = haskey(data, Symbol(force_key))
_has_virial(data; virial_key=:virial, kwargs...) = haskey(data, Symbol(virial_key))
