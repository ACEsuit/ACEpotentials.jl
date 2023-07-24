
using JuLIP: Atoms, energy, cutoff


at_dimer(r, z1, z0) = Atoms(X = [ SVector(0.0,0.0,0.0), SVector(r, 0.0, 0.0)], 
                            Z = [z0, z1], pbc = false, 
                            cell = [r+1 0 0; 0 1 0; 0 0 1])

at_trimer(r1, r2, θ, z0, z1, z2) = Atoms(X = [SVector(0.0, 0.0, 0.0), SVector(r1, 0.0, 0.0), SVector(r2 * cos(θ), r2 * sin(θ), 0.0)],
                            Z = [z0, z1, z2], pbc = false, #
                            cell = [ 1.0 + maximum([r1, r2]) 0.0 0.0;
                            0.0 (1 + r2) 0.0;
                            0.0 0.0 1])

function dimer_energy(IP, r, z1, z0)
   at = at_dimer(r, z1, z0)
   at1 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z1, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   at2 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   return energy(IP, at) - energy(IP, at1) - energy(IP, at2)
end

function trimer_energy(IP, r1, r2, θ, z0, z1, z2)
   at = at_trimer(r1, r2, θ, z0, z1, z2)
   at0 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0])
   at1 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z1, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0])
   at2 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z2, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0])
   dr1r2 = sqrt(r1 ^ 2 + r2 ^ 2 - 2 * r1 * r2 * cos(θ))
   return energy(IP, at) - 
         ACE1pack.dimer_energy(IP, r1, z0, z1) - ACE1pack.dimer_energy(IP, r2, z0, z2) - ACE1pack.dimer_energy(IP, dr1r2, z1, z2)
         - energy(IP, at0) - energy(IP, at1) - energy(IP, at2)
end

_cutoff(potential) = cutoff(potential)
_cutoff(potential::JuLIP.MLIPs.SumIP) = maximum(_cutoff.(potential.components))
_cutoff(potential::JuLIP.OneBody) = 0.0


"""
Generate a dictionary of dimer curves for a given potential. 
"""
function dimers(potential, elements; 
                rr = range(1e-3, _cutoff(potential), length=200), 
                minE = -1e10, maxE = 1e10, )
   zz = AtomicNumber.(elements)
   dimers = Dict() 
   for i = 1:length(zz), j = 1:i
      z1 = zz[i]
      z0 = zz[j]
      v01 = dimer_energy.(Ref(potential), rr, z1, z0)
      v01 = max.(min.(v01, maxE), minE)
      dimers[(z1, z0)] = (rr, v01)
   end
   return dimers
end

"""
Generate a dictionary of trimer curves for a given potential.
"""
function trimers(potential, elements; r1 = 0.5:3:_cutoff(potential), r2 = 0.5:3:_cutoff(potential),
                θ = range(deg2rad(-180), deg2rad(180), length = 200), 
                minE = -1e10, maxE = 1e10, )
   zz = AtomicNumber.(elements)
   trimers = Dict() 
   for i = 1:length(zz), j = 1:i, k = 1:j
      z0 = zz[i]
      z1 = zz[j]
      z2 = zz[k]
      for ri in r1
         for rj in r2
            v01 = trimer_energy.(Ref(potential), ri, rj, θ, z0, z1, z2)
            v01 = max.(min.(v01, maxE), minE)
            trimers[(z0, z1, z2, ri, rj)] = (θ, v01)
         end
      end
   end
   return trimers
end

"""
Produce a list of all r values that occur in the dataset 
"""
function get_rdf(data::AbstractVector{<: Atoms})
   R = Float64[] 
   for at in data 
      nlist = JuLIP.neighbourlist(at, r_cut)
      r = [ norm(rr) for (i, j, rr) in pairs(nlist) ] 
      append!(R, r)
   end
   sort!(R) 

   # todo - filter by 1/r^2 prob

   return R 
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

