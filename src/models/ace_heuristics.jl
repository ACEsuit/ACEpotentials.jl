

function ace_learnable_Rnlrzz(; 
               Dtot = nothing, 
               lmax = nothing, 
               elements = nothing, 
               spec = nothing, 
               rin0cuts = _default_rin0cuts(elements),
               transforms = agnesi_transform.(rin0cuts, 2, 2), 
               polys = Polynomials4ML.legendre_basis(Dtot+1), 
               envelopes = PolyEnvelope2sX(-1.0, 1.0, 2, 2)
               )
   if elements == nothing
      error("elements must be specified!")
   end
   if (spec == nothing) && (Dtot == nothing || lmax == nothing)
      error("Must specify either `spec` or `Dtot` and `lmax`")
   end

   zlist =_convert_zlist(elements)

   if spec == nothing
      spec = [ (n = n, l = l) for n = 1:(Dtot+1), l = 0:lmax 
                              if (n-1 + l) <= Dtot ]
   end

   maxn = maximum([ s.n for s in spec ])
   if maxn > length(polys)
      error("maxn > length of polynomial basis")
   end

   return LearnableRnlrzzBasis(zlist, polys, transforms, envelopes, rin0cuts, spec)
end 
