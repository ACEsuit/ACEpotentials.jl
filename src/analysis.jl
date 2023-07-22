
using JuLIP: Atoms, energy, cutoff


at_dimer(r, z1, z0) = Atoms(X = [ SVector(0.0,0.0,0.0), SVector(r, 0.0, 0.0)], 
                            Z = [z0, z1], pbc = false, 
                            cell = [r+1 0 0; 0 1 0; 0 0 1])


function dimer_energy(IP, r, z1, z0)
   at = at_dimer(r, z1, z0)
   at1 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z1, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   at2 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   return energy(IP, at) - energy(IP, at1) - energy(IP, at2)
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

