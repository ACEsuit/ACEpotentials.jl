# Shared test utilities for ACEpotentials ET backend tests

using AtomsBuilder, Unitful
using Random, Lux

"""
    rand_struct(n_repeat=(2,1,1); elements=[:Si => 0.5, :O => 0.5], rattle=0.2u"Å")

Create a randomized test structure for ACE testing.
"""
function rand_struct(n_repeat=(2,1,1); elements=[:Si => 0.5, :O => 0.5], rattle=0.2u"Å")
    sys = AtomsBuilder.bulk(:Si) * n_repeat
    AtomsBuilder.rattle!(sys, rattle)
    AtomsBuilder.randz!(sys, elements)
    return sys
end

"""
    setup_test_model(; elements=(:Si, :O), max_level=8, order=2, maxl=4, rcut=5.5)

Create a test ACE model with pair basis zeroed out.
Returns (model, ps, st, rcut).
"""
function setup_test_model(; elements=(:Si, :O), max_level=8, order=2, maxl=4, rcut=5.5,
                           rng=Random.MersenneTwister(1234))
    M = ACEpotentials.Models
    level = M.TotalDegree()

    rin0cuts = M._default_rin0cuts(elements)
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

    model = M.ace_model(; elements = elements, order = order,
                Ytype = :solid, level = level, max_level = max_level,
                maxl = maxl, pair_maxn = max_level,
                rin0cuts = rin0cuts,
                init_WB = :glorot_normal, init_Wpair = :glorot_normal)

    ps, st = Lux.setup(rng, model)

    # Zero out pair basis (not implemented in ET backend)
    for s in model.pairbasis.splines
        s.itp.itp.coefs[:] *= 0
    end

    return model, ps, st, rcut
end
