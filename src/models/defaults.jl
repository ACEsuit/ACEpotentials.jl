module DefaultHypers 

import YAML

# -------------- Bond-length heuristics

_lengthscales_path = joinpath(@__DIR__, "..", "..", "data",
                              "length_scales_VASP_auto_length_scales.yaml")
_lengthscales = YAML.load_file(_lengthscales_path)

bond_len(s::Symbol) = bond_len(AtomicNumber(s))
bond_len(z::AtomicNumber) = bond_len(convert(Int, z))

function bond_len(z::Integer)
   if haskey(_lengthscales, z)
      return _lengthscales[z]["bond_len"][1]
   elseif rnn(AtomicNumber(z)) > 0
      return rnn(AtomicNumber(z))
   end
   error("No typical bond length for atomic number $z is known. Please specify manually.")
end

bond_len(z1, z2) = (bond_len(z1) + bond_len(z2)) / 2


# -------------- 

end 