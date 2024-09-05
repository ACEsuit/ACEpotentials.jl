

using StaticArrays: SMatrix
import AtomsBase: ChemicalSpecies, atomic_number 

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
      return _convert_zlist( ntuple(i -> atomic_number( ChemicalSpecies(zlist[i]) ), 
                                    length(zlist) ) ) 
   elseif eltype(zlist) == ChemicalSpecies
      return tuple( atomic_number.(zlist)... )
   end 
   return ntuple(i -> convert(Int, zlist[i]), length(zlist))
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

