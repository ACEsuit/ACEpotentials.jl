
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

"""
Takes an object and converts it to an `SMatrix{NZ, NZ}` via the following rules: 
- if `obj` is already an `SMatrix{NZ, NZ}` then it just return `obj`
- if `obj` is an `AbstractMatrix` and `size(obj) == (NZ, NZ)` then it 
   converts it to an `SMatrix{NZ, NZ}` with the same entries.
- otherwise it generates an `SMatrix{NZ, NZ}` filled with the value `obj`.
"""
function _make_smatrix(obj, NZ) 
   if obj isa SMatrix{NZ, NZ}
      return obj
   end
   if obj isa AbstractMatrix && size(obj) == (NZ, NZ)
      return SMatrix{NZ, NZ}(obj)
   end
   if obj isa AbstractArray && size(obj) != (NZ, NZ) 
      error("`_make_smatrix` : if the input `obj` is an `AbstractArray` then it must be of size `(NZ, NZ)`")
   end
   return SMatrix{NZ, NZ}(fill(obj, (NZ, NZ)))
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

