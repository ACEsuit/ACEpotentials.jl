module ACE1compat

using NamedTupleTools, StaticArrays, Unitful
import ACEpotentials: Models 

using ACEpotentials.Models: ace_learnable_Rnlrzz

const _kw_defaults = Dict(
    :elements => nothing,
    :order => nothing,
    :totaldegree => nothing,
    :wL => 1.5,
    :rin => 0.0,
    :r0 => :bondlen,
    :rcut => (:bondlen, 2.5),
    :transform => (:agnesi, 2, 4),
    :envelope => (:x, 2, 2),
    :rbasis => :legendre,
    :pair_rin => :rin,
    :pair_rcut => :rcut,
    :pair_degree => :totaldegree,
    :pair_transform => (:agnesi, 1, 3),
    :pair_basis => :legendre,
    :pair_envelope => (:r, 2),
    :Eref => missing,
    :variable_cutoffs => false
)


function bond_len(z1, z2)
    return 1.0  
end

function _get_elements(kwargs)
    if kwargs[:elements] === nothing
        error("Elements must be specified in the arguments.")
    end
    return collect(kwargs[:elements])
end

function _get_all_rcut(kwargs; _rcut = nothing)
    elements = _get_elements(kwargs)

    if _rcut === nothing
        _rcut = kwargs[:rcut]
    end

    if _rcut isa Number
        return _rcut
    elseif _rcut isa Dict
        return Dict((s1, s2) => _get_rcut(kwargs, s1, s2; _rcut = _rcut) for s1 in elements, s2 in elements)
    else
        error("Unsupported rcut format.")
    end
end

function _ace1_rin0cuts(kwargs; rcutkey = :rcut)
    elements = _get_elements(kwargs)
    rcut = _get_all_rcut(kwargs; _rcut = kwargs[rcutkey])
    if rcut isa Number
        cutoffs = Dict([ (s1, s2) => (0.0, rcut) for s1 in elements, s2 in elements ]...)
    else
        cutoffs = Dict([ (s1, s2) => (0.0, rcut[(s1, s2)]) for s1 in elements, s2 in elements ]...)
    end
    
    rin0cuts = _rin0cuts_rcut(elements, cutoffs)
    return rin0cuts
end

function _rin0cuts_rcut(zlist, cutoffs::Dict)
    function rin0cut(zi, zj)
        r0 = bond_len(zi, zj)
        rin, rcut = cutoffs[zi, zj]
        return (rin = rin, r0 = r0, rcut = rcut)
    end
    NZ = length(zlist)
    return SMatrix{NZ, NZ}([rin0cut(zi, zj) for zi in zlist, zj in zlist])
end

function _clean_args(kwargs)
    dargs = Dict{Symbol, Any}()
    for key in keys(_kw_defaults)
        dargs[key] = get(kwargs, key, _kw_defaults[key])
    end

    if dargs[:pair_rcut] == :rcut
        dargs[:pair_rcut] = dargs[:rcut]
    end

    return namedtuple(dargs)
end

function _get_order(kwargs)
    return get(kwargs, :order, get(kwargs, :bodyorder, 2) - 1)
end

function _get_degrees(kwargs)
    deg = kwargs[:totaldegree]
    cor_order = _get_order(kwargs)

    maxlevels = deg isa Number ? [deg for _ in 1:cor_order] : deg

    wL = kwargs[:wL]
    NZ = length(kwargs[:elements])
    return Models.TotalDegree(1.0 * NZ, 1 / wL), maxlevels
end

function _get_r0(kwargs, z1, z2)
    return kwargs[:r0] == :bondlen ? bond_len(z1, z2) : kwargs[:r0]
end

function _get_all_r0(kwargs)
    elements = kwargs[:elements]
    return Dict((s1, s2) => _get_r0(kwargs, s1, s2) for s1 in elements, s2 in elements)
end

function _get_rcut(kwargs, s1, s2)
    _rcut = kwargs[:rcut]
    return _rcut isa Tuple && _rcut[1] == :bondlen ? _rcut[2] * _get_r0(kwargs, s1, s2) : _rcut
end

function _transform(kwargs)
    elements = kwargs[:elements]

    if kwargs[:transform][1] == :agnesi
        rin0cuts = _ace1_rin0cuts(kwargs)
        return rin0cuts  
    else
        error("Unsupported transform.")
    end
end

function _get_Rnl_spec(kwargs, maxdeg = maximum(kwargs[:totaldegree]))
    wL = kwargs[:wL]
    NZ = length(kwargs[:elements])

    lvl = 1.0 * NZ / wL
    specs = [(n = n, l = l) for n in 1:maxdeg, l in 0:maxdeg-1]

    if maximum(n for (n, l) in specs) > maxdeg
        error("Configured maxn exceeds the available polynomial basis length.")
    end

    return specs
end

function _radial_basis(kwargs)
    trans_ace = _transform(kwargs)
    rin0cuts = _ace1_rin0cuts(kwargs)
    Rnl_spec = _get_Rnl_spec(kwargs)
    polys = (:jacobi, 2.0, 2.0)
    return ace_learnable_Rnlrzz(
        spec = Rnl_spec,
        maxq = maximum(b.n for b in Rnl_spec),
        elements = kwargs[:elements],
        rin0cuts = rin0cuts,
        transforms = trans_ace,
        polys = polys,
        Winit = :onehot
    )
end

function _pair_basis(kwargs)
    maxq = ceil(Int, kwargs[:pair_degree] == :totaldegree ? maximum(kwargs[:totaldegree]) : kwargs[:pair_degree])
    rin0cuts = _ace1_rin0cuts(kwargs; rcutkey = :pair_rcut)
    trans_pair = _transform(kwargs; transform = kwargs[:pair_transform], rcutkey = :pair_rcut)
    envelope = kwargs[:pair_envelope][1] == :r ? (:r_ace1, kwargs[:pair_envelope][2]) : kwargs[:pair_envelope]

    return ace_learnable_Rnlrzz(
        spec = [(n = n, l = 0) for n in 1:maxq * length(kwargs[:elements])],
        maxq = maxq,
        elements = kwargs[:elements],
        rin0cuts = rin0cuts,
        transforms = trans_pair,
        envelopes = envelope,
        polys = :legendre,
        Winit = :onehot
    )
end

function ace1_model(; kwargs...)
    kwargs = _clean_args(kwargs)
    elements = kwargs[:elements]
    cor_order = _get_order(kwargs)
   # rbasis = _radial_basis(kwargs)
    #pairbasis = _pair_basis(kwargs)
    lvl, maxlvl = _get_degrees(kwargs)

    rin0cuts = Models._default_rin0cuts(elements) #; rcutfactor = 2.29167)

    model = Models.ace_model(;
        elements = elements, 
        order = cor_order,               # correlation order 
        Ytype = :spherical,              # solid vs spherical harmonics
        #E0s = ismissing(kwargs[:Eref]) ? nothing : Dict([key => val * u"eV" for (key, val) in kwargs[:Eref]]...),
        level = Models.TotalDegree(),     # how to calculate the weights to give to a basis function
        max_level = maxlvl[1],     # maximum level of the basis functions
        pair_maxn = maxlvl[1],     # maximum number of basis functions for the pair potential 
        init_WB = :zeros,            # how to initialize the ACE basis parmeters
        init_Wpair = "linear",         # how to initialize the pair potential parameters
        init_Wradial = :linear, 
        pair_transform = (:agnesi, 1, 3), 
        pair_learnable = false, 
        rin0cuts = rin0cuts, 
    )
    return model
end

end


######################################################

using .ACE1compat
using JSON
working_dir = "/home/vekondra/ACEpotentials.jl/scripts"
using Pkg
Pkg.activate(joinpath(working_dir, "."))

function create_namedtuple(dict)
   return NamedTuple{Tuple(Symbol.(keys(dict)))}(values(dict))
end

function _sanitize_arg(arg)
   if isa(arg, Vector)  
       return _sanitize_arg.(tuple(arg...))
   elseif isa(arg, String)
       return Symbol(arg)
   else
       return arg
   end
end

function _sanitize_dict(dict)
   return Dict(Symbol(key) => _sanitize_arg(dict[key]) for key in keys(dict))
end

# function make_acemodel(model_dict::Dict)
#    model_nt = _sanitize_dict(model_dict)
#    return ACE1compat.ace1_model(; model_nt...)
# end

function make_acemodel(model_dict::Dict)
    fitting_params = load_dict(joinpath(working_dir, "fitting_params.json"))
    if fitting_params["model"]["model_name"] == "ACE1"
        model_nt = _sanitize_dict(model_dict)
        return ACE1compat.ace1_model(; model_nt...)
    else
        error("Unknown model: $(fitting_params["model"]["model_name"]). This function only supports 'ACE1'.")
    end
end

function make_solver(solver_dict::Dict)
    if solver_dict["name"] == "BLR"
        params_nt = _sanitize_dict(solver_dict["param"])
        return ACEfit.BLR(; params_nt...)
    else
        error("Not implemented.")
    end
end

function make_prior(model, prior_dict::Dict)
    if prior_dict["name"] === "algebraic"
        return ACEpotentials.Models.algebraic_smoothness_prior(model.basis; p = prior_dict["param"])
    else
        error("Not implemented.")
    end
end

function _check_args(args_nt::NamedTuple)
    # make sure it has the required keys
end

function fallback_default()
    # fill in args by default and raise info for those 
end
