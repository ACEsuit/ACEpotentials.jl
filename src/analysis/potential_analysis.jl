
using JuLIP: Atoms, energy, cutoff

__2z(z::AtomicNumber) = z 
__2z(s::Symbol) = AtomicNumber(s)
__2z(s::String) = AtomicNumber(Symbol(s))


"""
`function at_dimer(r, z1, z0)` : generates a dimer with separation `r` and 
atomic numbers `z1` and `z0`.  (can also use symbols or strings)
"""
at_dimer(r, z1, z0) = Atoms(X = [ SVector(0.0,0.0,0.0), SVector(r, 0.0, 0.0)], 
                            Z = __2z.([z0, z1]), pbc = false, 
                            cell = [r+1 0 0; 0 1 0; 0 0 1])

"""
`function at_trimer(r1, r2, θ, z0, z1, z2)` : generates a trimer with
separations `r1` and `r2`, angle `θ` and atomic numbers `z0`, `z1` and `z2` 
(can also use symbols or strings),  where `z0` is the species of the central 
atom, `z1` at distance `r1` and `z2` at distance `r2`.
"""
at_trimer(r1, r2, θ, z0, z1, z2) = Atoms(X = [SVector(0.0, 0.0, 0.0), SVector(r1, 0.0, 0.0), SVector(r2 * cos(θ), r2 * sin(θ), 0.0)],
                            Z = [z0, z1, z2], pbc = false, #
                            cell = [ 1.0 + maximum([r1, r2]) 0.0 0.0;
                            0.0 (1 + r2) 0.0;
                            0.0 0.0 1])

"""
`function atom_energy(IP, z0)` : energy of an isolated atom
"""                            
function atom_energy(IP, z0)
   at = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], 
              pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0])
   return energy(IP, at)                  
end

"""
`function dimer_energy(pot, r, z1, z0)` : energy of a dimer 
with separation `r` and atomic numbers `z1` and `z0` using the potential `pot`; 
subtracting the 1-body contributions. 
"""
function dimer_energy(IP, r, z1, z0)
   at = at_dimer(r, z1, z0)
   return energy(IP, at) - atom_energy(IP, z0) - atom_energy(IP, z1)
end

"""
`function trimer_energy(IP, r1, r2, θ, z0, z1, z2)` : computes the energy of a
trimer, subtracting the 2-body and 1-body contributions.
"""
function trimer_energy(IP, r1, r2, θ, z0, z1, z2)
   at = at_trimer(r1, r2, θ, z0, z1, z2)
   dr1r2 = sqrt(r1 ^ 2 + r2 ^ 2 - 2 * r1 * r2 * cos(θ))
   return ( energy(IP, at) 
            - dimer_energy(IP, r1, z1, z0) 
            - dimer_energy(IP, r2, z2, z0) 
            - dimer_energy(IP, dr1r2, z1, z2)
            - atom_energy(IP, z0) 
            - atom_energy(IP, z1) 
            - atom_energy(IP, z2) )
end

# TODO: this needs a PR to JuLIP 
_cutoff(potential) = cutoff(potential)
_cutoff(potential::JuLIP.MLIPs.SumIP) = maximum(_cutoff.(potential.components))
_cutoff(potential::JuLIP.OneBody) = 0.1   # not sure how 0.0 will behave


"""
Generate a dictionary of dimer curves for a given potential. 
"""
function dimers(potential, elements; 
                rr = range(1e-3, _cutoff(potential), length=200), 
                minE = -1e5, maxE = 1e5, )
   zz = AtomicNumber.(elements)
   dimers = Dict() 
   for i = 1:length(zz), j = 1:i
      z1 = zz[i]
      z0 = zz[j]
      v01 = dimer_energy.(Ref(potential), rr, z1, z0)
      v01 = max.(min.(v01, maxE), minE)
      dimers[(z1, z0)] = (rr, v01)
   end
   copy_zz_sym!(dimers)
   return dimers
end

"""
Generate a dictionary of trimer curves for a given potential.
"""
function trimers(potential, elements, r1, r2; 
                θ = range(-pi, pi, length = 200), 
                minE = -1e5, maxE = 1e5, )
   zz = AtomicNumber.(elements)
   trimers = Dict() 
   for i = 1:length(zz), j = 1:i, k = 1:j
      z0 = zz[i]
      z1 = zz[j]
      z2 = zz[k]
      v01 = trimer_energy.(Ref(potential), r1, r2, θ, z0, z1, z2)
      v01 = max.(min.(v01, maxE), minE)
      trimers[(z0, z1, z2)] = (θ, v01)
   end
   copy_zz_sym!(trimers)
   return trimers
end

# function trimers(potential, elements; r1 = 0.5:3:_cutoff(potential), r2 = 0.5:3:_cutoff(potential),
#                 θ = range(deg2rad(-180), deg2rad(180), length = 200), 
#                 minE = -1e10, maxE = 1e10, )
#    zz = AtomicNumber.(elements)
#    trimers = Dict() 
#    for i = 1:length(zz), j = 1:i, k = 1:j
#       z0 = zz[i]
#       z1 = zz[j]
#       z2 = zz[k]
#       for ri in r1
#          for rj in r2
#             v01 = trimer_energy.(Ref(potential), ri, rj, θ, z0, z1, z2)
#             v01 = max.(min.(v01, maxE), minE)
#             trimers[(z0, z1, z2, ri, rj)] = (θ, v01)
#          end
#       end
#    end
#    return trimers
# end





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
   if aa == :auto 
      rcut = _cutoff(pot) 
      aa = range(0.0, rcut, length=npoints)
   end
                
   set_calculator!(at0, pot)
   variablecell!(at0)
   minimise!(at0)

   E0 = energy(pot, at0) / length(at0)

   atref = at0 * (8, 1, 1)
   Cref = copy(Matrix(atref.cell))
   Xref = copy(positions(atref))

   function decoh_energy(_a_)
      C = copy(Cref); C[1] += _a_
      at = deepcopy(atref)
      set_cell!(at, C)
      set_positions!(at, Xref)
      return energy(pot, at) - length(at) * E0
   end

   E = decoh_energy.(aa)
   dE = [[0.0]; (E[2:end] - E[1:end-1])/(aa[2]-aa[1])]

   return E, dE 
end   



function get_transforms(model::ACE1x.ACE1Model)
   mb_transforms = Dict() 
   pair_transforms = Dict()

   pair_basis = model.basis.BB[1] 
   @assert typeof(pair_basis) <: PolyPairBasis
   for iz0 = 1:JuLIP.numz(pair_basis), iz = 1:JuLIP.numz(pair_basis)
      Pr = pair_basis.J[iz0,iz]
      z = JuLIP.i2z(pair_basis, iz)
      z0 = JuLIP.i2z(pair_basis, iz0)
      s = chemical_symbol(z)
      s0 = chemical_symbol(z0)
      mb_transforms[(z0, z)] = Pr.trans
      mb_transforms[(s0, s)] = Pr.trans
   end

   ace_basis = model.basis.BB[2] 
   @assert typeof(ace_basis) <: RPIBasis
   basis1p = ace_basis.pibasis.basis1p 
   mtrans = basis1p.J.trans
   @assert mtrans isa ACE1.Transforms.MultiTransform
   for iz0 = 1:JuLIP.numz(basis1p), iz = 1:JuLIP.numz(basis1p)
      z = JuLIP.i2z(basis1p, iz)
      z0 = JuLIP.i2z(basis1p, iz0)
      s = chemical_symbol(z)
      s0 = chemical_symbol(z0)
      mb_transforms[(z0, z)] = mtrans.transforms[iz0,iz]
      mb_transforms[(s0, s)] = mtrans.transforms[iz0,iz] 
   end

   return mb_transforms, pair_transforms 
end
