
using JuLIP: Atoms, energy, cutoff

function dimer_energy(IP, r, z1, z0)
   at = at_dimer(r, z1, z0)
   at1 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z1, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   at2 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   return energy(IP, at) - energy(IP, at1) - energy(IP, at2)
end


_cutoff(potential) = cutoff(potential)
_cutoff(potential::JuLIP.MLIPs.SumIP) = maximum(_cutoff.(potential.components))
_cutoff(potential::JuLIP.OneBody) = 0.0

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
