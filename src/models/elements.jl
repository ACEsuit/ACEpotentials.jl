
using JuLIP: AtomicNumber
using StaticArrays: SMatrix
import ACE1x


_i2z(obj, i::Integer) = obj._i2z[i] 

_get_nz(obj) = length(obj._i2z)

function _z2i(obj, Z)
   for i_Z = 1:length(obj._i2z)
      if obj._i2z[i_Z] == Z
         return i_Z
      end
   end
   error("_z2i : Z = $Z not found in obj._i2z")
   return -1 # never reached
end

# convert AtomicNumber -> Int is already defined in JuLIP 
# we also want Symbol -> Int, but this would be terrible type piracy!
# so intead we make it a case distinction inside the _convert_zlist. 
# not elegant but works for now. 

function _convert_zlist(zlist) 
   if eltype(zlist) == Symbol 
      return ntuple(i -> convert(Int, AtomicNumber(zlist[i])), length(zlist))
   end
   return ntuple(i -> convert(Int, zlist[i]), length(zlist))
end

function _default_rin0cuts(zlist; rinfactor = 0.0, rcutfactor = 2.5)
   function rin0cut(zi, zj) 
      r0 = ACE1x.get_r0(zi, zj)
      return (rin = r0 * rinfactor, r0 = r0, rcut = r0 * rcutfactor)
   end
   NZ = length(zlist)
   return SMatrix{NZ, NZ}([ rin0cut(zi, zj) for zi in zlist, zj in zlist ])
end


# a one-hot embedding for the z variable. 
# function embed_z(ace, Rs, Zs)
#    TF = eltype(eltype(Rs))
#    Ez = acquire!(ace.pool, :Ez, (length(Zs), length(ace.rbasis)), TF)
#    fill!(Ez, 0)
#    for (j, z) in enumerate(Zs)
#       iz = _z2i(ace.rbasis, z)
#       Ez[j, iz] = 1
#    end
#    return Ez 
# end

