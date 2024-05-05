


using Pkg; Pkg.activate(".");
using TestEnv; TestEnv.activate();

using ACEpotentials
import Polynomials4ML
P4ML = Polynomials4ML
M = ACEpotentials.Models

using Random, LuxCore
rng = Random.MersenneTwister(1234)

##

# LearnableRnlrzzBasis(
#             zlist, polys, transforms, envelopes, rincut, spec::Vector{T_NL_TUPLE}; 
#             weights=nothing, meta=Dict{String, Any}()) = 
#    LeanrableRnlBasis(_convert_zlist(zlist), polys, 
#                      _auto_trans(transforms, length(zlist)), 
#                      _auto_envel(envelopes, length(zlist)), 
#                      _auto_rincut(rincut, length(zlist)), 
#                      _auto_weights(weights, length(zlist)), 
#                      meta)

Dtot = 5
lmax = 3 
elements = (:Si, :O)
zlist = M._convert_zlist(elements)
rin0cuts = M._default_rin0cuts(elements)
transforms = M.agnesi_transform.(rin0cuts, 2, 2)
polys = P4ML.legendre_basis(Dtot+1)
envelopes = M.PolyEnvelope2sX(-1.0, 1.0, 2, 2)
spec = [ (n = n, l = l) for n = 1:(Dtot+1), l = 0:lmax if (n-1 + l) <= Dtot ]

basis = M.LearnableRnlrzzBasis(zlist, polys, transforms, envelopes, rin0cuts, spec)
ps, st = LuxCore.setup(rng, basis)

r = 3.0 
Zi = zlist[1] 
Zj = zlist[2]
Rnl, st1 = basis(r, Zi, Zj, ps, st)
