
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Export

import ACE1, JuLIP
using ACE1.RPI: BasicPSH1pBasis, PSH1pBasisFcn
using ACE1: PIBasis, PIBasisFcn, PIPotential
using ACE1.OrthPolys: TransformedPolys
using ACE1: rand_radial, cutoff, numz
using JuLIP: energy, bulk, i2z, z2i, chemical_symbol
using ACE1.PairPotentials: RepulsiveCore, PolyPairPot

function export_ace(fname::AbstractString, args...;  kwargs...)
   fptr = open(fname; write=true)
   export_ace(fptr, args...; kwargs...)
   close(fptr)
end


function export_ace(fptr::IOStream, Pr::TransformedPolys; ntests=0, kwargs...)
   p = Pr.trans.p
   r0 = Pr.trans.r0
   xr = Pr.J.tr
   xl = Pr.J.tl
   pr = Pr.J.pr
   pl = Pr.J.pl
   rcut = cutoff(Pr)
   maxn = length(Pr)

   println(fptr, "transform parameters: p=$(p) r0=$(r0)")
   println(fptr, "cutoff parameters: rcut=$rcut xl=$xl xr=$xr pl=$pl pr=$pr")
   println(fptr, "recursion coefficients: maxn=$(maxn)")
   for n = 1:maxn
      println(fptr, " $(Pr.J.A[n]) $(Pr.J.B[n]) $(Pr.J.C[n])")
   end

   # save some tests
   println(fptr, "tests: ntests=$(ntests)")
   for itest = 1:ntests
      r = ACE1.rand_radial(Pr)
      P = ACE1.evaluate(Pr, r)
      dP = ACE1.evaluate_d(Pr, r)
      println(fptr, " r=$(r)")
      for n = 1:length(P)
         println(fptr, " $(P[n]) $(dP[n])")
      end
   end
end

function export_ace(fptr::IOStream, basis::ACE1.Splines.RadialSplines)

   # these are hardcoded in PACE
   ntot = 10000 # number of bins for lookup table
   nlut = 10000 # number of nodes for lookup table

   # PACE lookup table. the final dimension holds the coefficients
   #   c0 + c1*(r-rn) + c2*(r-rn)^2 + c3*(r-rn)^3
   #   where rn is a bin lower bound
   lookupTable = zeros(ntot+1, length(basis.splines), 4)

   # TODO: populate the lookup table using something analagous to
   # ACE1.Splines.export_splines

end

function export_ace(fptr::IOStream, Vpair::PolyPairPot; kwargs...)
   println(fptr, "begin polypairpot")
   # this exports everything except the coefficients
   export_ace(fptr, Vpair.basis.J; kwargs...)
   println(fptr, "coefficients")
   c = Vpair.coeffs
   for n = 1:length(c)
      println(fptr, "$(c[n])")
   end
   println(fptr, "end polypairpot")
end

function export_ace(fptr::IOStream, Vrep::RepulsiveCore; kwargs...)
   println(fptr, "begin repulsive potential")
   # export the pair potential part
   export_ace(fptr, Vrep.Vout; kwargs...)
   println(fptr, "spline parameters")
   println(fptr, "   e_0 + B  exp(-A*(r/ri-1)) * (ri/r)")
   Vin = Vrep.Vin[1]
   println(fptr, "ri=$(Vin.ri)")
   println(fptr, "e0=$(Vin.e0)")
   println(fptr, "A=$(Vin.A)")
   println(fptr, "B=$(Vin.B)")
   println(fptr, "end repulsive potential")
end

function export_ace(fptr::IOStream, V::PIPotential, Vpair=nothing, V0=nothing; kwargs...)
   @assert numz(V) == 1
   inner = V.pibasis.inner[1]
   coeffs = V.coeffs[1]
   # sort the basis functions into groups the way the ace evaluator wants it
   groups = _basis_groups(inner, coeffs)
   lmax = maximum(maximum(g["l"]) for g in groups)

   sym = chemical_symbol(i2z(V, 1))
   haspair = (Vpair == nothing) ? "f" : "t"

   # header
   println(fptr, "nelements=1")
   println(fptr, "elements: $sym")
   println(fptr, "")
   println(fptr, "lmax=$lmax")
   println(fptr, "")
   println(fptr, "2 FS parameters:  1.000000 1.000000")
   println(fptr, "core energy-cutoff parameters: 100000.000000000000000000 1.000000000000000000")

   # write E0 values
   if V0 == nothing
      println(fptr, "E0: 0.0")
   else
      println(fptr, "E0: $(V0(sym))")
   end

   println(fptr, "")
   println(fptr, "radbasename=ACE1.jl.Basic")
   println(fptr, "haspair: $(haspair)")
   println(fptr, "")

   # export_ace the radial basis
   #export_ace(fptr, V.pibasis.basis1p.J; kwargs...)
   # TODO: improve this hack
   export_ace(fptr, ACE1.Splines.RadialSplines(V.pibasis.basis1p.J))
   println(fptr, "")

   if haspair == "t"
      export_ace(fptr, Vpair)
      println(fptr, "")
   end

   # header
   rankmax = maximum(length(g["l"]) for g in groups)
   println(fptr, "rankmax=$rankmax")
   println(fptr, "ndensitymax=1")
   println(fptr, "")

   # header for basis list pair contributions
   println(fptr, "num_c_tilde_max=$(length(groups))")
   num_ms_combinations_max = maximum( length(g["M"]) for g in groups )
   println(fptr, "num_ms_combinations_max=$(num_ms_combinations_max)")

   # write the pair basis groups
   total_basis_size_rank1 = sum( (length(g["l"]) ==  1) for g in groups )
   println(fptr, "total_basis_size_rank1: $(total_basis_size_rank1)")
   for i = 1:total_basis_size_rank1
      g = groups[i]
      _write_group(fptr, g)
   end

   # write the rest
   total_basis_size = length(groups) - total_basis_size_rank1
   println(fptr, "total_basis_size: $(total_basis_size)")
   for i = (total_basis_size_rank1+1):length(groups)
      g = groups[i]
      _write_group(fptr, g)
   end

end

function _write_group(fptr, g)
   order = length(g["l"])
   println(fptr, "ctilde_basis_func: rank=$(order) ndens=1 mu0=0 mu=(" * " 0 "^order * ")")
   println(fptr, "n=(" * prod(" $(ni) " for ni in g["n"]) * ")")
   println(fptr, "l=(" * prod(" $(li) " for li in g["l"]) * ")")
   println(fptr, "num_ms=$(length(g["M"]))")
   for (m, c) in zip(g["M"], g["C"])
      c_ace = c / (4*π)^(order/2)
      println(fptr, "<" * prod(" $(mi) " for mi in m) * ">:  $(c_ace)")
   end
end



function export_ace_tests(fname::AbstractString, V, ntests = 1;
                          nrepeat = 3, pert=0.05,
                          s = JuLIP.chemical_symbol(i2z(V, 1)))
   at = bulk(s, cubic=true) * nrepeat
   JuLIP.set_pbc!(at, false)
   r0 = JuLIP.rnn(s)
   for n = 1:ntests
      JuLIP.rattle!(at, pert * r0)
      E = energy(V, at)
      _write_test(fname * "_$n.dat", JuLIP.positions(at), E)
   end
   return at
end

function _write_test(fname, X, E)
   fptr = open(fname; write=true)
   println(fptr, "E = $E")
   println(fptr, "natoms = $(length(X))")
   println(fptr, "# type x y z")
   for n = 1:length(X)
      println(fptr, "0 $(X[n][1]) $(X[n][2]) $(X[n][3])")
   end
   close(fptr)
end



function _basis_groups(inner, coeffs)
   NL = []
   M = []
   C = []
   for b in keys(inner.b2iAA)
      if coeffs[ inner.b2iAA[b] ] != 0
         push!(NL, ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps] ))
         push!(M, [b1.m for b1 in b.oneps])
         push!(C, coeffs[ inner.b2iAA[b] ])
      end
   end
   ords = length.(M)
   perm = sortperm(ords)
   NL = NL[perm]
   M = M[perm]
   C = C[perm]
   @assert issorted(length.(M))
   bgrps = []
   alldone = fill(false, length(NL))
   for i = 1:length(NL)
      if alldone[i]; continue; end
      nl = NL[i]
      Inl = findall(NL .== Ref(nl))
      alldone[Inl] .= true
      Mnl = M[Inl]
      Cnl = C[Inl]
      pnl = sortperm(Mnl)
      Mnl = Mnl[pnl]
      Cnl = Cnl[pnl]
      push!(bgrps, Dict("n" => nl[1], "l" => nl[2],
                        "M" => Mnl, "C" => Cnl))
   end
   return bgrps
end


end
