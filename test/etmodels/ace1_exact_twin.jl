# The "exact twin" of an `ace1_model`: the same construction as src/ace1_compat.jl
# (`_radial_basis`, `_pair_basis`, `ace1_model`) minus its two `splinify()` calls.
#
# WHY.  `ace1_model` splinifies both radial bases.  The ETModels conversion
# (`convert2et`, `convertpair`) and the LAMMPS/trim exporter need the polynomial
# `LearnableRnlrzzBasis` the splines were tabulated from, so a model built with
# `ace1_model` is converted / exported through this unsplined twin.  Everything else --
# `Ytype = :spherical`, the Jacobi(2pin, 2pcut) radial polynomials, the per-species-pair
# `ACE1_PolyEnvelope1sR` pair envelope, the A/AA/B specification -- is what `ace1_model`
# builds, and `ace1_twin_matches` below checks that the twin's specifications are
# identical to those of the splinified model, so the two take the same `WB` / `Wpair`.
#
# Shared by test/etmodels/test_etpair_ace1.jl (the ETModels conversion) and
# export/test/test_ace1_export.jl (the exporter).  Keep the two in step by keeping them
# both on this file.

using ACEpotentials, Lux, Random, Unitful

const _C1 = ACEpotentials.ACE1compat
const _M1 = ACEpotentials.Models

"""
    ace1_exact_twin(; kwargs...) -> ACEModel

The unsplinified twin of `ace1_model(; kwargs...)`.  Only the options the ACE1compat
`:legendre` radial and pair bases accept are supported (the defaults); anything else
raises, as it does in `ace1_model`.
"""
function ace1_exact_twin(; kwargs...)
   kw = _C1._clean_args(kwargs)
   @assert !kw[:ZBL] "ace1_exact_twin: ZBL is not supported (the ETModels conversion has no ZBL term)"
   @assert kw[:rbasis] == :legendre && kw[:pair_basis] == :legendre
   @assert kw[:envelope] == (:x, 2, 2)
   elements = _C1._get_elements(kw)
   NZ = length(elements)

   # --- many-body radial basis: _radial_basis without splinify
   rin0cuts = _C1._ace1_rin0cuts(kw)
   Rnl_spec = _C1._get_Rnl_spec(kw)
   pin, pcut = kw[:envelope][2], kw[:envelope][3]
   rbasis = _M1.ace_learnable_Rnlrzz(; spec = Rnl_spec,
                  maxq = maximum(b.n for b in Rnl_spec), elements = elements,
                  rin0cuts = rin0cuts, transforms = _C1._transform(kw),
                  polys = (:jacobi, Float64(2 * pin), Float64(2 * pcut)),
                  Winit = :onehot)

   # --- pair basis: _pair_basis without splinify
   @assert kw[:pair_degree] == :totaldegree
   maxq = ceil(Int, maximum(kw[:totaldegree]))
   envelope = kw[:pair_envelope]
   @assert envelope isa Tuple && envelope[1] == :r "ace1_exact_twin: only the default (:r, p) pair envelope"
   pairbasis = _M1.ace_learnable_Rnlrzz(; spec = [(n = n, l = 0) for n in 1:maxq*NZ],
                  maxq = maxq, elements = elements,
                  rin0cuts = _C1._ace1_rin0cuts(kw; rcutkey = :pair_rcut),
                  transforms = _C1._transform(kw, transform = kw[:pair_transform],
                                              rcutkey = :pair_rcut),
                  envelopes = (:r_ace1, envelope[2]), polys = :legendre,
                  Winit = :onehot)

   # --- the model: as ace1_model
   lvl, maxlvl = _C1._get_degrees(kw)
   E0s = ismissing(kw[:Eref]) ? nothing :
         Dict([key => val * u"eV" for (key, val) in kw[:Eref]]...)
   return _M1.ace_model(; elements = elements, order = _C1._get_order(kw),
                  Ytype = :spherical, ZBL = false, E0s = E0s,
                  rbasis = rbasis, pair_basis = pairbasis,
                  rin0cuts = rbasis.rin0cuts, level = lvl, max_level = maxlvl,
                  init_WB = :zeros)
end

"""
    ace1_twin_matches(twin, spl) -> Bool

Whether the twin and the splinified `ace1_model` (`spl`, an ACEModel) have the same
basis: A2B maps, the AA specification and the radial / pair specifications.
"""
function ace1_twin_matches(twin, spl)
   return twin.rbasis.spec == spl.rbasis.spec &&
          twin.pairbasis.spec == spl.pairbasis.spec &&
          twin.tensor.meta["𝔸spec"] == spl.tensor.meta["𝔸spec"] &&
          all(Matrix(a) == Matrix(b) for (a, b) in zip(twin.tensor.A2Bmaps, spl.tensor.A2Bmaps)) &&
          twin.rbasis._i2z == spl.rbasis._i2z && twin.pairbasis._i2z == spl.pairbasis._i2z
end
