
using AtomsBase: FastSystem, ChemicalSpecies, Unitful
using AtomsCalculatorsUtilities.SitePotentials: cutoff_radius 
using AtomsCalculators: potential_energy, energy_unit 
using StaticArrays
using AtomsBuilder 
using LinearAlgebra: norm 

__2z(z::Integer) = z 
__2z(s::Symbol) = atomic_number(s)
__2z(s::String) = atomic_number(Symbol(s))
__2z(s::ChemicalSpecies) = atomic_number(s)


"""
`function at_dimer(r, z1, z0)` : generates a dimer with separation `r` and 
atomic numbers `z1` and `z0`.  (can also use symbols or strings)
"""
function at_dimer(r, z1, z0) 
   uL = unit(r)
   s = ustrip(r)  
   box = ( SA[s+1, 0.0, 0.0]*uL, SA[0.0, 1.0, 0.0]*uL, SA[0.0, 0.0, 1.0]*uL )
   pbc = (false, false, false)
   positions = [ SA[0.0,0.0,0.0] * uL, SA[s, 0.0, 0.0] * uL ]
   Z = __2z.([z0, z1])
   M = fill( 1.0 * u"u", length(Z) )
   return FastSystem(box, pbc, positions, Z, M)
end

"""
`function at_trimer(r1, r2, θ, z0, z1, z2)` : generates a trimer with
separations `r1` and `r2`, angle `θ` and atomic numbers `z0`, `z1` and `z2` 
(can also use symbols or strings),  where `z0` is the species of the central 
atom, `z1` at distance `r1` and `z2` at distance `r2`.
"""
function at_trimer(r1, r2, θ, z0, z1, z2) 
   s1 = ustrip(r1)
   s2 = ustrip(r2)
   uL = unit(r1)
   @assert uL == unit(r2)
   box = (   SA[1.0 + maximum([s1, s2]), 0.0, 0.0]*uL, 
             SA[0.0, 1.0 + s2, 0.0]*uL, 
             SA[0.0, 0.0, 1.0]*uL  )
   pbc = (false, false, false)
   positions = [ SVector(0.0, 0.0, 0.0) * uL, 
                 SVector(s1, 0.0, 0.0) * uL, 
                 SVector(s2 * cos(θ), s2 * sin(θ), 0.0) * uL ]
   Z = __2z.([z0, z1, z2])
   M = fill( 1.0 * u"u", length(Z) )
   return FastSystem(box, pbc, positions, Z, M)
end                 


"""
`function atom_energy(IP, z0)` : energy of an isolated atom
"""                            
function atom_energy(IP, z0)
   box = ( SA[1.0, 0.0, 0.0]*u"Å", SA[0.0, 1.0, 0.0]*u"Å", SA[0.0, 0.0, 1.0]*u"Å" )
   pbc = (false, false, false)
   positions = [ SA[0.0,0.0,0.0] * u"Å" ]
   Z = [__2z(z0), ] 
   M = fill( 1.0 * u"u", length(Z) )
   sys = FastSystem(box, pbc, positions, Z, M)
   return potential_energy(sys, IP)
end

"""
`function dimer_energy(pot, r, z1, z0)` : energy of a dimer 
with separation `r` and atomic numbers `z1` and `z0` using the potential `pot`; 
subtracting the 1-body contributions. 
"""
function dimer_energy(IP, r, z1, z0)
   sys = at_dimer(r, z1, z0)
   return potential_energy(sys, IP) - atom_energy(IP, z0) - atom_energy(IP, z1)
end

"""
`function trimer_energy(IP, r1, r2, θ, z0, z1, z2)` : computes the energy of a
trimer, subtracting the 2-body and 1-body contributions.
"""
function trimer_energy(IP, r1, r2, θ, z0, z1, z2)
   sys = at_trimer(r1, r2, θ, z0, z1, z2)
   dr1r2 = sqrt(r1 ^ 2 + r2 ^ 2 - 2 * r1 * r2 * cos(θ))
   return ( potential_energy(sys, IP) 
            - dimer_energy(IP, r1, z1, z0) 
            - dimer_energy(IP, r2, z2, z0) 
            - dimer_energy(IP, dr1r2, z1, z2)
            - atom_energy(IP, z0) 
            - atom_energy(IP, z1) 
            - atom_energy(IP, z2) )
end

# function copy_zz_sym!(D::AbstractDict)
#    _zz = collect(keys(D))
#    for z12 in _zz
#       sym12 = Symbol.( ChemicalSpecies.(z12) ) 
#       D[sym12] = D[z12]
#    end
# end

"""
`dimers(potential, elements; kwargs...)` : 
Generate a dictionary of dimer curves for a given potential. 
* `potential` : potential to use to evaluate energy
* `elements` : list of chemical species, symbols for which the dimers are to be computed

The function returns a dictionary `Ddim` such that `D[(s1, s2)]` contains
pairs or arrays `(rr, E)` which can be plotted `plot(rr, E)`. 
"""
function dimers(potential, elements; 
                rmax = cutoff_radius(potential),
                rmin = 1e-4 * rmax, 
                rr = range(rmin, rmax, length=200), 
                minE = -1e5 * energy_unit(potential), 
                maxE = 1e5  * energy_unit(potential), )
   dimers = Dict() 
   for i = 1:length(elements), j = 1:i
      z1 = __2z(elements[i])
      z0 = __2z(elements[j])
      v01 = dimer_energy.(Ref(potential), rr, z1, z0)
      v01 = max.(min.(v01, maxE), minE)
      dimers[(z1, z0)] = (rr, v01)
   end
   copy_zz_sym!(dimers)
   return dimers
end

"""
`trimers(potential, elements, r1, r2; kwargs...)` : 
Generate a dictionary of trimer curves for a given potential. 

* `potential` : potential to use to evaluate energy 
* `elements` : list of chemical species, symbols for which the trimers are to be computed
* `r1, r2` : distance between the central atom and the first, second neighbour

The function returns a dictionary `Dtri` such that `D[(s1, s2, s3)]` contains 
pairs or arrays `(θ, E)` which can be plotted `plot(θ, E)`. 
"""
function trimers(potential, elements, r1, r2; 
                θ = range(-pi, pi, length = 200), 
                minE = -1e5 * energy_unit(potential), 
                maxE = 1e5  * energy_unit(potential), )
   trimers = Dict() 
   for i = 1:length(elements), j = 1:i, k = 1:j
      z0 = elements[i]
      z1 = elements[j]
      z2 = elements[k]
      v01 = trimer_energy.(Ref(potential), r1, r2, θ, z0, z1, z2)
      v01 = max.(min.(v01, maxE), minE)
      trimers[(z0, z1, z2)] = (θ, v01)
   end
   copy_zz_sym!(trimers)
   return trimers
end





"""
Generate a decohesion curve for testing the smoothness of a potential. 
Arguments:
- `at0` : unit cell 
- `pot` : potential implementing `energy`
Keyword Arguments: 
- `dim = 1` : dimension into which to expand
- `mult = 10` : multiplicative factor for expanding the cell in dim direction
- `aa = :auto` : array of stretch values of the lattice parameter to use
- `npoints = 100` : number of points to use in the stretch array (for auto aa)
"""
function decohesion_curve(at0, pot; 
               dim = 1, mult = 10, 
               aa = :auto, npoints = 100)

   # TODO: technically we should first relax the cell; for the moment we will 
   #       make the user responsible for this!!! But we should come back. 
   # set_calculator!(at0, pot)
   # variablecell!(at0)
   # minimise!(at0)

   E0 = potential_energy(at0, pot) / length(at0)

   atref = at0 * ntuple(i -> (i == dim ? mult : 1), 3) # (mult, 1, 1) if dim == 1
   Cref = deepcopy(cell_vectors(atref)) 
   Xref = deepcopy(position(atref, :))

   if aa == :auto 
      rcut = cutoff_radius(pot)
      uL = unit(rcut)
      a_max = rcut / norm(Cref[dim]) 
      aa = range(0.0, a_max, length=npoints)
   end


   function decoh_energy(_a_)
      C = [ deepcopy(Cref)... ]; C[dim] *= (1+_a_)
      at = AbstractSystem(atref; cell_vectors = tuple(C...))
      return potential_energy(at, pot) - length(at) * E0
   end

   E = decoh_energy.(aa)
   dE = [[0.0]; (E[2:end] - E[1:end-1])/(aa[2]-aa[1])]
   rr = aa * norm(Cref[dim]) 

   return rr, E, dE 
end   


# some additional imports needed only locally 

import ACEpotentials 
import ACEpotentials: ACEPotential 
import ACEpotentials.Models: ACEModel

function get_transforms(model::ACEPotential{<: ACEModel})
   mb_transforms = Dict() 
   pair_transforms = Dict()
   _i2z = ACEpotentials.Models._i2z

   NZ = ACEpotentials.Models._get_nz(model.model)
   for iz0 = 1:NZ, iz = 1:NZ 
      z0 = _i2z(model.model, iz0); z = _i2z(model.model, iz)
      mb_transforms[(z0, z)] = model.model.rbasis.transforms[iz0,iz]
      pair_transforms[(z0, z)] = model.model.pairbasis.transforms[iz0,iz]
   end

   copy_zz_sym!(mb_transforms)
   copy_zz_sym!(pair_transforms)

   return mb_transforms, pair_transforms 
end
