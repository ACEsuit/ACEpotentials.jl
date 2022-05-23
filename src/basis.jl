
# ------------------------------------------
#   ACE Basis  

import ACE1.PairPotentials: PolyPairBasis
import ACE1.Transforms: PolyTransform, MultiTransform


export basis_params, degree_params, transform_params


"""
`basis_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct one of the basis. 
All parameters are passed as keyword argument and the kind of 
parameters required depend on "type". 

## ACE (RPI) basis 
Returns a dictionary containing the complete set of parameters 
required to construct an ACE basis (`RPIBasis`). All parameters 
are passed as keyword argument. If no default is given then 
the argument is required. 

### Parameters
* `type = "ace"`
* `species` : single species or list of species (mandatory)
* `N` : correlation order, positive integer (mandatory)
* `maxdeg` : maximum degree, positive real number (note the precise 
notion of degree is specified by further parameters) (mandatory)
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `radial = radial_basis_params(; r0 = r0)` : one-particle basis 
parameters; cf `?basis_params` of type "radial" for details 
* `transform = transform_params(; r0 = r0)` : distance transform 
parameters; cf `?transform_params()` for details
* `degree = degree_params()` : class of sparse polynomial degree 
to select the basis; see `?degree_params` for details 

## Pair basis 
Returns a dictionary containing the complete set of parameters 
required to construct an pair basis (`PolyPairBasis`). All 
parameters are passed as keyword argument. 

### Parameters
* `type = "pair"`
* `species` : single species or list of species (mandatory)
* `maxdeg` : maximum degree, positive real number (note the precise 
notion of degree is specified by further parameters) (mandatory)
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `rcut = 5.0`: outer cuttoff, Å 
* `rin = 0.0`: inner cuttoff, Å 
* `pcut = 2`: outer cutoff parameter
* `pin = 0`: inner cutoff parameter
* `transform = transform_params(; r0 = r0)` : distance transform 
parameters; cf `?transform_params()` for details

## Radial basis of ACE
Returns a dictionary containing the complete set of parameters 
required to construct radial basis for ACE. All parameters are 
passed as keyword argument. 

### Parameters
* `type = "radal"`
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `rcut = 5.0`: outer cuttoff, Å 
* `rin = 0.0`: inner cuttoff, Å 
* `pcut = 2`: outer cutoff parameter
* `pin = 0`: inner cutoff parameter
"""
function basis_params(;
      type = nothing, 
      kwargs...)
      @assert haskey(_bases, type) "type $(type) not found among available types of basis ($(keys(_bases)))"
      return _bases[type][2](; kwargs...)
end

function generate_basis(params::Dict)
      @assert params["type"] != "radal" 
      params = copy(params)
      basis_constructor = _bases[params["type"]][1]
      delete!(params, "type")
      return basis_constructor(params)
end


# ------------------------------------------
#  ace basis 

"""
`ace_basis_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct an ACE basis (`RPIBasis`). 
All parameters are passed as keyword argument. If no default is given then 
the argument is required. 

### Parameters
* `species` : single species or list of species (mandatory)
* `N` : correlation order, positive integer (mandatory)
* `maxdeg` : maximum degree, positive real number (note the precise notion of 
degree is specified by further parameters) (mandatory)
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `radial = radial_basis_params(; r0 = r0)` : one-particle basis parameters; 
cf `?radial_basis_params` for details 
* `transform = transform_params(; r0 = r0)` : distance transform parameters; 
cf `?transform_params()` for details
* `degree = degree_params()` : class of sparse polynomial degree to select 
the basis; see `?degree_params` for details 
"""
function ace_basis_params(; 
      species = nothing, 
      N::Integer = nothing, 
      maxdeg = nothing, 
      r0 = 2.5, 
      radial = radial_basis_params(; r0 = r0), 
      transform = transform_params(; r0 = r0), 
      degree = degree_params(),
      type = "ace"
   )
    
   @assert !isnothing(species) "`species` is mandatory for `ace_basis_params`"
   @assert isinteger(N) "correlation order `N` must be a positive integer"
   @assert N > 0 "correlation order `N` must be a positive integer"
   @assert isreal(maxdeg) "Maximum polynomial degree `maxdeg` must be a real positive number"
   @assert maxdeg > 0 "Maximum polynomial degree `maxdeg` must be a real positive number"
   @assert isreal(r0) "`r0` must be a real positive number "
   @assert r0 > 0 "`r0` must be a real positive number "
   @assert type == "ace" "`type` must be set to \"ace\" for `ace_basis_params`"

   if !haskey(radial, "type")
      radial = convert(Dict{String, Any}, radial)
      radial["type"] = "radial"
   end

   return Dict( 
         "type" => "ace",
         "species" => _species_to_params(species), 
         "N" => N, 
         "maxdeg" => maxdeg, 
         "radial" => radial, 
         "transform" => transform, 
         "degree" => degree
         )
end

"""Returns ACE1.Utils.rpi_basis """
function generate_ace_basis(params::Dict)
   species = _params_to_species(params["species"])
   trans = generate_transform(params["transform"])
   D = generate_degree(params["degree"])
   maxdeg = params["maxdeg"]
   radial = generate_radial_basis(params["radial"], D, maxdeg, species, trans)
   return ACE1.Utils.rpi_basis(; 
            species = species, 
            N = params["N"], 
            trans = trans, 
            D = D, 
            maxdeg = maxdeg, 
            rbasis = radial, 
         )
end


# ------------------------------------------
#  pair basis 


"""
`pair_basis_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct an pair basis (`PolyPairBasis`). 
All parameters are passed as keyword argument. 

### Parameters
* `species` : single species or list of species (mandatory)
* `maxdeg` : maximum degree, positive real number (note the precise notion of degree 
is specified by further parameters) (mandatory)
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `rcut = 5.0`: outer cuttoff, Å 
* `rin = 0.0`: inner cuttoff, Å 
* `pcut = 2`: outer cutoff parameter
* `pin = 0`: inner cutoff parameter
* `transform = transform_params(; r0 = r0)` : distance transform parameters; 
cf `?transform_params()` for details
"""
function pair_basis_params(;
      species = nothing,
      maxdeg = nothing, 
      r0 = 2.5,
      rcut = 5.0,
      rin = 0.0,
      pcut = 2, 
      pin = 0,
      transform = transform_params(; r0=r0),
      type = "pair",
      )

      @assert !isnothing(species) "`species` is mandatory for `ace_basis_params`"
      @assert isreal(maxdeg) "Maximum polynomial degree `maxdeg` must be a real positive number"
      @assert maxdeg > 0 "Maximum polynomial degree `maxdeg` must be a real positive number"
      @assert isreal(r0) "`r0` must be a real positive number "
      @assert r0 > 0 "`r0` must be a real positive number "
      @assert type == "pair" "`type` must be set to \"pair\" for `pair_basis_params`"

      return Dict(
            "type" => "pair",
            "species" => _species_to_params(species),
            "maxdeg" => maxdeg,
            "rcut" => rcut,
            "rin" => rin,
            "pcut" => pcut,
            "pin" => pin,
            "transform" => transform)
end

""" Returns PolyPairBasis """
function generate_pair_basis(params::Dict)
      species = _params_to_species(params["species"])
      trans = generate_transform(params["transform"])
      rad_basis = transformed_jacobi(params["maxdeg"], trans, params)
      return PolyPairBasis(rad_basis, species)

end

# ------------------------------------------
#  rad_basis 

"""
`radial_basis_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct radial basis for ACE. 
All parameters are passed as keyword argument. 

### Parameters
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `rcut = 5.0`: outer cuttoff, Å 
* `rin = 0.0`: inner cuttoff, Å 
* `pcut = 2`: outer cutoff parameter
* `pin = 0`: inner cutoff parameter
"""
function radial_basis_params(; 
      r0 = 2.5,
      rcut = 5.0,
      rin = 0.5 * r0,
      pcut = 2,
      pin = 2,
      type = "radial")

      @assert isreal(r0) "`r0` must be a real positive number "
      @assert r0 > 0 "`r0` must be a real positive number "
      @assert type == "radial" "`type` must be set to \"radial\" for `radial_basis_params`"

      return Dict(
            "type" => "radial",
            "rcut" => rcut, 
            "rin" => rin, 
            "pcut" => pcut, 
            "pin" => pin )
end   

function generate_radial_basis(params::Dict, D, maxdeg, species, trans)
   maxn = ACE1.RPI.get_maxn(D, maxdeg, species)
   return transformed_jacobi(maxn, trans, params)
end


# ------------------------------------------
#  basis helper functions 


_bases = Dict("pair" => (generate_pair_basis, pair_basis_params),  
              "ace" => (generate_ace_basis, ace_basis_params),
              "radial" => (nothing, radial_basis_params))


transformed_jacobi(maxn::Integer, trans::MultiTransform, params::Dict) = 
      OrthPolys.transformed_jacobi(maxn, trans; pcut = params["pcut"], pin = params["pin"])

transformed_jacobi(maxn::Integer, trans::PolyTransform, params::Dict) =
      OrthPolys.transformed_jacobi(maxn, trans, params["rcut"], params["rin"]; 
                         pcut = params["pcut"], pin = params["pin"])


# ------------------------------------------
#  degree 

"""
`degree_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a specification
for polynomial degree. All parameters are passed as keyword argument 
and the kind of parameters required depend on "type". 

## `SparsePSHDegree` 
Returns a dictionary containing the complete set of parameters required 
to construct `ACE1.RPI.SparsePSHDegree`. See `?SparsePSHDegree`.

### Parameters
* `type = "sparse"`
* `wL = 1.5`
* `csp = 1.0` 
* `chc = 0.0`
* `chc = 0.0`
* `ahc = 0.0`
* `bhc = 0.0`
* `p = 1.0`

## `SparsePSHDegreeM`
Returns a dictionary containing the complete set of parameters required 
to construct `ACE1.RPI.SparsePSHDegree`. Also see `?SparsePSHDegreeM`. 

NB `maxdeg` of ACE basis (`RPIBasis`) has to be set to `1.0`.

### Parameters
* `Dd` : Dictionary specifying max degrees (mandatory)
* `Dn = Dict("default" => 1.0)` : Dictionary specifying weights for degree 
of radial basis functions (n)
* `Dl = Dict("default" => 1.5)` : Dictionary specifying weights for degree 
of angular basis functions (l)

Each dictionary should have a "default" entry. In addition, different degrees 
or weights can be specified for each correlation order and/or correlation 
order-species combination. For example 

```            
"Dd" => Dict(
      "default" => 10,
      3 => 9,
      (4, "C") => 8,
      (4, "H") => 0)
```

in combination with N=4 and maxdeg=1.0, will set maximum polyonmial degree on 
N=1 and N=2 functions to 10, to 9 for N=3 functions and will only allow 
N=4 basis functions on carbon atoms, up to polynomial degree 8. 

"""
function degree_params(;
      type = "sparse", 
      kwargs...)
      @assert haskey(_degrees, type)
      return _degrees[type][2](; kwargs...)
end

function generate_degree(params::Dict)
      @assert haskey(_degrees, params["type"])
      # we ignore p for `SparsePSHDegree`, for now
      if params["type"] == "sparse" && haskey(params, "p")
            delete!(params, "p")
      end
      degree_measure = _degrees[params["type"]][1]
      kwargs = Dict([Symbol(key) => val for (key, val) in params]...)
      delete!(kwargs, :type)
      return degree_measure(; kwargs...)
end

"""
`sparse_degree_params(; kwargs...)`: returns a dictionary containing the 
complete set of parameters required to construct `ACE1.RPI.SparsePSHDegree`.
See `?SparsePSHDegree`.

### Parameters
* `wL = 1.5`
* `csp = 1.0` 
* `chc = 0.0`
* `chc = 0.0`
* `ahc = 0.0`
* `bhc = 0.0`
* `p = 1.0`

NB `p = 1` is current ignored, but we put it in so we can experiment later 
with `p = 2`, `p = inf`. 
""" 
function sparse_degree_params(; 
      wL::Real = 1.5, 
      csp::Real = 1.0, 
      chc::Real = 0.0, 
      ahc::Real = 0.0, 
      bhc::Real = 0.0, 
      p::Real = 1.0 )

   @assert wL > 0 
   @assert csp >= 0 && chc >= 0 
   @assert csp > 0 || chc > 0 
   @assert ahc >= 0 && bhc >= 0
   @assert p == 1
   return Dict( "type" => "sparse", 
                "wL" => wL, 
                "csp" => csp, 
                "chc" => chc, 
                "ahc" => ahc, 
                "bhc" => bhc, 
                "p" => p )
end 

"""
`sparse_degree_M_params(;kwargs...)`: Returns a dictionary containing the 
complete set of parameters required to construct `ACE1.RPI.SparsePSHDegree`. 
Also see `?SparsePSHDegreeM`. 

NB `maxdeg` of ACE basis (`RPIBasis`) has to be set to `1.0`.

### Parameters
* `Dd` : Dictionary specifying max degrees (mandatory)
* `Dn = Dict("default" => 1.0)` : Dictionary specifying weights for degree 
of radial basis functions (n)
* `Dl = Dict("default" => 1.5)` : Dictionary specifying weights for degree 
of angular basis functions (l)

Each dictionary should have a "default" entry. In addition, different degrees 
or weights can be specified for each correlation order and/or correlation 
order-species combination. For example 

```            
"Dd" => Dict(
      "default" => 10,
      3 => 9,
      (4, "C") => 8,
      (4, "H") => 0)
```

in combination with N=4 and maxdeg=1.0, will set maximum polyonmial degree on 
N=1 and N=2 functions to 10, to 9 for N=3 functions and will only allow 
N=4 basis functions on carbon atoms, up to polynomial degree 8. 
"""
function sparse_degree_M_params(;
      Dd::Dict = nothing,
      Dn::Dict = Dict("default" => 1.0),
      Dl::Dict = Dict("default" => 1.5))

      @assert !isnothing(Dd) "`Dd`` is a mandatory."

      return Dict(
            "type" => "sparseM",
            "Dd" => _AtomicNumber_to_params(Dd),
            "Dn" => _AtomicNumber_to_params(Dn), 
            "Dl" => _AtomicNumber_to_params(Dl))
end

SparsePSHDegreeM(; Dn::Dict, Dl::Dict, Dd::Dict) = 
      ACE1.RPI.SparsePSHDegreeM(_params_to_AtomicNumber(Dn), 
                                _params_to_AtomicNumber(Dl), 
                                _params_to_AtomicNumber(Dd))

_degrees = Dict(
      "sparse" => (ACE1.RPI.SparsePSHDegree, sparse_degree_params),
      "sparseM" => (SparsePSHDegreeM, sparse_degree_M_params))


# ------------------------------------------
#  transform 
#  this is a little more interesting since there are quite a 
#  few options. 

"""
`transform_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct one of the transforms. 
All parameters are passed as keyword argument and the kind of 
parameters required depend on "type".


## Polynomial transform 
Returns a dictionary containing the complete set of parameters required 
to construct `ACE1.Transforms.PolyTransform``. All parameters are passed
 as keyword argument. Also see `?PolyTransform`

Implements the distance transform

```math
   x(r) = \\Big(\\frac{1 + r_0}{1 + r}\\Big)^p
```

### Parameters
* `type = "polynomial"`
* `p = 2` 
* `r0 = 2.5`


## Multitransform
Returns a dictionary containing the complete set of parameters required 
to construct `ACE.Transform.multitransform`. All parameters are passed 
as keyword argument. 

### Parameters
* `transforms` : dictionary specifying transforms for each species pair. Can be
given per-pair (i.e. only for `("element1", "element2")` and not for 
`("element2", "element1")`) or can be different for `("element1", "element2")` and 
`("element2", "element1")`. For example
```
transforms = Dict(
      ("C", "C") => Dict("type"=> "polynomial"),
      ("C", "H") => Dict("type"=> "polynomial"),
      ("H", "H") => Dict("type" => "polynomial"))
```

* `rin`, `rcut`: values for inner and outer cutoffs, alternative to `cutoffs`
* `cutoffs` : dictionary specifying inner and outer cutoffs for each element pair
(either symmetrically or non-symmetrically). Alternative to `rin` & `rcut`. 
For example
```
cutoffs => Dict(
      ("C", "C") => (1.1, 4.5),
      ("C", "H") => (0.9, 4.5),
      ("H", "H") => (1.23, 4.5)),
``` 


## identity
`IdTransform_params(;)` : returns `Dict("type" => "identity")`,
needed to construct `ACE1.Transforms.IdTransform`.  
"""
function transform_params(; 
      type = "polynomial",
      kwargs...)
   @assert haskey(_transforms, type)
   return _transforms[type][2](; kwargs...)
end

"""
`PolyTransform_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct `ACE1.Transforms.PolyTransform``.
All parameters are passed as keyword argument. Also see `?PolyTransform`

Implements the distance transform

```math
   x(r) = \\Big(\\frac{1 + r_0}{1 + r}\\Big)^p
```

### Parameters
* `p = 2` 
* `r0 = 2.5`
"""
function PolyTransform_params(; p = 2, r0 = 2.5)
   @assert isreal(p) "`p`` must be a real positive number"
   @assert p > 0 "`p`` must be a real positive number"
   @assert isreal(r0) "`r0` must be a real positive number "
   @assert r0 > 0 "`r0` must be a real positive number "

   return Dict("type" => "polynomial", 
               "p" => p, 
               "r0" => r0)
end

"""
`multitransform_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct `ACE.Transform.multitransform`.
All parameters are passed as keyword argument. 

### Parameters
* `transforms` : dictionary specifying transforms for each species pair. Can be
given per-pair (i.e. only for `("element1", "element2")` and not for 
`("element2", "element1")`) or can be different for `("element1", "element2")` and 
`("element2", "element1")`. For example
```
transforms = Dict(
      ("C", "C") => Dict("type"=> "polynomial"),
      ("C", "H") => Dict("type"=> "polynomial"),
      ("H", "H") => Dict("type" => "polynomial"))
```

* `rin`, `rcut`: values for inner and outer cutoffs, alternative to `cutoffs`
* `cutoffs` : dictionary specifying inner and outer cutoffs for each element pair
(either symmetrically or non-symmetrically). Alternative to `rin` & `rcut`. 
For example
```
cutoffs => Dict(
      ("C", "C") => (1.1, 4.5),
      ("C", "H") => (0.9, 4.5),
      ("H", "H") => (1.23, 4.5)),
``` 
"""
function multitransform_params(; 
      transforms = nothing,
      rin = nothing,
      rcut = nothing,
      cutoffs=nothing)

      @assert !isnothing(transforms) "`transforms` must be specified."
      @assert (!isnothing(rin) && !isnothing(rcut)) || !isnothing(cutoffs) "Either `rin` & `rcut` or `cutoffs` must be given."

      return Dict("type" => "multitransform",
                  "transforms" => _species_to_params(transforms),
                  "rin" => rin,
                  "rcut" => rcut,
                  "cutoffs" => _species_to_params(cutoffs))
end

function generate_multitransform(; transforms, rin = nothing, rcut = nothing, cutoffs = nothing)
      transforms = _params_to_species(transforms)
      transforms = Dict(key => generate_transform(params) for (key, params) in transforms)
      cutoffs = _params_to_species(cutoffs)
      return ACE1.Transforms.multitransform(transforms, rin = rin, rcut = rcut, cutoffs = cutoffs)
end

"""
`IdTransform_params(;)` : returns `Dict("type" => "identity")`,
needed to construct `ACE1.Transforms.IdTransform`.  
"""
IdTransform_params(;) = Dict("type" => "identity")

function generate_transform(params::Dict)
   @assert haskey(_transforms, params["type"])
   TTransform = _transforms[params["type"]][1]
   kwargs = Dict([Symbol(key) => val for (key, val) in params]...)
   delete!(kwargs, :type)
   return TTransform(; kwargs...)
end

# In this dictionary we "register" all the transforms for which we have 
# supplied an interface. At the moment I've done it just for one of them 
# others can introduce more. The key is a string the specifies the key 
# user supplies for the `type` parameter. The value is a tuple containing 
# the corresponding transform type and function that generates the defaul 
# parameters 
_transforms = Dict(
      "polynomial" => (ACE1.Transforms.PolyTransform, PolyTransform_params),
      "multitransform" => (generate_multitransform, multitransform_params),
      "identity" => (ACE1.Transforms.IdTransform, IdTransform_params)
)


# ------------------------------------------
# helper functions

# -- Symbol to string
_species_to_params(species::Union{Symbol, AbstractString}) = 
      [ string(species), ] 

_species_to_params(species::Union{Tuple, AbstractArray}) = 
      collect( string.(species) )

# accept tuples of Symbol or String for dictionary key
# values can be anything
_species_to_params(dict::Dict{Tuple{Tsym, Tsym}, Tval}) where Tsym <: Union{Symbol, AbstractString} where Tval <: Any = 
      Dict(Tuple(_species_to_params(key)) => val for (key, val) in dict)

_species_to_params(dict::Nothing) = nothing


# -- String to Symbol
_params_to_species(species::Union{AbstractArray{T}, Tuple{T, T}}) where T <: AbstractString  = 
      Symbol.(species)

_params_to_species(dict::Dict{Tuple{Tsym, Tsym}, Tval}) where Tsym <: AbstractString where Tval <: Any = 
      Dict(Tuple(_params_to_species(d)) => val for (d, val) in dict)

_params_to_species(dict::Nothing) = nothing


function _AtomicNumber_to_params(dict)
      new_dict = Dict()
      for (key, val) in dict
            if typeof(key) <: Tuple
                 key = Tuple(typeof(entry) <: AtomicNumber ? string(chemical_symbol(entry)) : entry for entry in key)
            end
            new_dict[key] = val
      end
      return new_dict
end

function _params_to_AtomicNumber(dict)
      new_dict = Dict()
      for (key, val) in dict
            if typeof(key) <: Tuple
                  key = Tuple(typeof(entry) <: AbstractString && length(entry) == 1 ? 
                              AtomicNumber(Symbol(entry)) : entry for entry in key)
            end
            new_dict[key] = val
      end
      return new_dict
end