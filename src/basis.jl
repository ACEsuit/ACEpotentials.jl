
# ------------------------------------------
#   ACE Basis  

export rpi_basis_params, pair_basis_params, degree_params, radbasis_params, transform_params


# ------------------------------------------
#  rpi basis 

"""
`rpi_basis_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct an ACE basis (`RPIBasis`). 
All parameters are passed as keyword argument. If no default is given then 
the argument is required. 

### Parameters
* `species` : single species or list of species 
* `N` : correlation order 
* `maxdeg` : maximum degree  (note the precise notion of degree is specified by further parameters)
* `r0 = 2.5` : rough estimate for nearest neighbour distance
* `basis1p = basis1p_params(; r0 = r0)` : one-particle basis parameters; cf `?basis1p_params` for details 
* `transform = transform_params(; r0 = r0)` : distance transform parameters; cf `?transform_params()` for details
* `degree = degree_params()` : class of sparse polynomial degree to select the basis; see `?degree_params` for details 
"""
function rpi_basis_params(; 
      species = nothing, 
      N::Integer = nothing, 
      maxdeg = nothing, 
      r0 = 2.5, 
      radbasis = radbasis_params(; rin = 0.5 * r0, pin = 2), 
      transform = transform_params(; r0 = r0), 
      degree = degree_params()
   )
   # TODO: replace assert statements with user-friendly error messages
   @assert !isnothing(species)
   @assert isinteger(N) 
   @assert N > 0 
   @assert isreal(maxdeg) 
   @assert maxdeg > 0 
   @assert isreal(r0)
   @assert r0 > 0 
   return Dict( 
         "species" => _species_to_params(species), 
         "N" => N, 
         "maxdeg" => maxdeg, 
         "radbasis" => radbasis, 
         "transform" => transform, 
         "degree" => degree
         )
end

function generate_rpi_basis(params::Dict)
   species = _params_to_species(params["species"])
   trans = generate_transform(params["transform"])
   D = generate_degree(params["degree"])
   maxdeg = params["maxdeg"]
   radbasis = generate_rpi_radbasis(params["radbasis"], D, maxdeg, species, trans)
   return ACE1.Utils.rpi_basis(; 
            species = species, 
            N = params["N"], 
            # r0 = r0,    
            trans = trans, 
            D = D, 
            maxdeg = maxdeg, 
            rbasis = radbasis, 
         )
end


# ------------------------------------------
#  pair basis 


"""TODO add documentation"""
function pair_basis_params(;
      species = nothing,
      maxdeg = nothing, 
      r0 = 2.5,
      radbasis = radbasis_params(; rin = 0.0, pin = 0),
      transform = transform_params(; r0=r0),
      )
      # TODO: replace asserts with something friendlier
      @assert !isnothing(species)
      @assert isreal(maxdeg)
      @assert maxdeg > 0
      @assert isreal(r0)
      @assert r0 > 0
      return Dict(
            "species" => _species_to_params(species),
            "maxdeg" => maxdeg,
            "radbasis" => radbasis,
            "transform" => transform
      )
end

"""TODO add documentation"""
function generate_pair_basis(params::Dict)
      species = _params_to_species(params["species"])
      trans = generate_transform(params["transform"])
      rb_params = params["radbasis"]
      radbasis = transformed_jacobi(params["maxdeg"], 
                                    trans, 
                                    rb_params["rcut"],
                                    rb_params["rin"];
                                    pcut = rb_params["pcut"],
                                    pin = rb_params["pin"])
      return ACE1.Utils.pair_basis(;
            species = species,
            trans = trans, 
            rbasis = radbasis,
      )
end




_species_to_params(species::Union{Symbol, AbstractString}) = 
      [ string(species), ] 

_species_to_params(species::Union{Tuple, AbstractArray}) = 
      collect( string.(species) )

_params_to_species(species::AbstractArray{<: AbstractString}) = 
      Symbol.(species)


# ------------------------------------------
#  degree 

# ENH: polynomial degree for each correlation order

"""
TODO: needs docs 

* `p = 1` is current ignored, but we put it in so we can experiment later 
with `p = 2`, `p = inf`. 
""" 
function degree_params(; 
      type::String = "sparse", 
      wL::Real = 1.5, 
      csp::Real = 1.0, 
      chc::Real = 0.0, 
      ahc::Real = 0.0, 
      bhc::Real = 0.0, 
      p::Real = 1.0 )

   @assert type in [ "sparse", ]
   @assert wL > 0 
   @assert csp >= 0 && chc >= 0 
   @assert csp > 0 || chc > 0 
   @assert ahc >= 0 && bhc >= 0
   @assert p == 1
   return Dict( "type" => type, 
                "wL" => wL, 
                "csp" => csp, 
                "chc" => chc, 
                "ahc" => ahc, 
                "bhc" => bhc, 
                "p" => p )
end 

function generate_degree(params::Dict) 
   @assert params["type"] == "sparse"
   return SparsePSHDegree(;  wL = params["wL"], 
                            csp = params["csp"], 
                            chc = params["chc"], 
                            ahc = params["ahc"],
                            bhc = params["bhc"] 
                          )
end

# ------------------------------------------
#  radbasis 

"""
TODO: needs docs 
""" 
function radbasis_params(; 
      rin = nothing, 
      pin = nothing,
      rcut = 5.0,
      pcut = 2)

      @assert !isnothing(rin)
      @assert !isnothing(pin)

   # TODO put in similar checks 
   return Dict( "rcut" => rcut, 
                "rin" => rin, 
                "pcut" => pcut, 
                "pin" => pin 
               )
end   

function generate_rpi_radbasis(params::Dict, D, maxdeg, species, trans)
   maxn = ACE1.RPI.get_maxn(D, maxdeg, species)
   return transformed_jacobi(maxn, trans, params["rcut"], params["rin"];
                             pcut = params["pcut"], pin = params["pin"] )
end


# ------------------------------------------
#  transform 
#  this is a little more interesting since there are quite a 
#  few options. 

# ENH: add multitransform

"""
TODO: needs docs
"""
function transform_params(; 
      type = "polynomial",
      kwargs... 
   )
   @assert haskey(_transforms, type)
   return _transforms[type][2](; kwargs...)
end


function PolyTransform_params(; p = 2, r0 = 2.5)
   @assert isreal(p) 
   @assert p > 0 
   @assert isreal(r0)
   @assert r0 > 0 
   return Dict("type" => "polynomial", 
               "p" => p, 
               "r0" => r0)
end


function generate_transform(params::Dict)
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
_transforms = Dict( "polynomial" => (ACE1.Transforms.PolyTransform, PolyTransform_params) )

