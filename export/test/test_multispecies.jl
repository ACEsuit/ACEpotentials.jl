#=
Multi-species export tests -- NZ = 3, gated at 1e-12.

WAS "both radial modes".  The `:hermite_spline` mode was removed (evidence and the
maintainer's answer: `export/bench/FINDINGS_parity.md` §7), so the three Hermite cases here
went with it: the uniform-cutoff Hermite gate against the splinified stack, the
per-pair-cutoff refusal, and the `@test_throws BoundsError` that locked the upstream
`EquivariantTensors._spl_grid` crash.  The pair-index coverage those cases carried is not
lost -- it moved onto the `:uniform` model's `:polynomial` export, which is now gated like
the `:asym` one.

WHY THESE MODELS.  Every per-pair table in a generated model is addressed by a species-pair
index, and there are two candidate conventions:

    ordered    k = (iz - 1) * NZ + jz                  (NZ^2 entries, centre species first)
    symmetric  k = symidx(min, max, NZ)                (NZ(NZ+1)/2 entries)

The *model* uses both: `ET.SelectLinL` weights (many-body `RBASIS_W`, the pair `PAIR_C`) are
per ORDERED pair, while the Agnesi transform parameters are stored per SYMMETRIC pair (`ETModels._convert_agnesi` loops `for i = 1:NZ, j = i:NZ` and
selects with `ET.catcat2idx_sym`).  The generator resolves the symmetric storage into ordered
tables at export time, so at RUNTIME there is exactly one convention: `pair_idx(iz, jz)`.

No single-species model can tell the two conventions apart (both give 1), and the Cantor
fixture cannot either: it was fitted with a scalar `r0 = 2.54` and a single `rcut`, so all of
its stored Agnesi tuples are byte-identical and no permutation of them is observable.  The
models below are the first ones on this branch that *can* discriminate:

  * their six symmetric Agnesi tuples are mutually distinct (asserted), so a wrong
    ordered -> symmetric expansion changes the numbers;
  * `init_Wradial = :glorot_normal` makes `RBASIS_W[k] != RBASIS_W[k']` for transposed pairs
    (asserted), so a symmetric index into an ordered table is observable;
  * the pair basis carries a per-symmetric-pair `r0`, so the pair term's six tuples are
    distinct too.

READ THIS BEFORE CHANGING THE CUTOFFS.  Two variants are needed, and the reason is an
upstream limitation, not a preference:

  `:asym`     many-body `rin0cuts[i,j].rcut = 4.6 + 0.2i + 0.3j` -- genuinely per-pair
              cutoffs, the harder case for the ordered/symmetric expansion.
  `:uniform`  one cutoff for every pair, with a per-symmetric-pair `r0` instead.  Equally
              discriminating for the index convention, and the shape a real multi-element
              model usually has.  Both are exported and gated; keeping the second is what
              stops the per-pair-table assertions from having only one subject.

  (Both variants existed because only `:uniform` could be splinified -- the `:asym` model
  threw a `BoundsError` inside upstream `_spl_grid`.  With the spline export gone that is no
  longer why there are two, but two discriminating models with different cutoff structures
  are worth keeping on their own account.)

REFERENCES AND TOLERANCES (metric definitions are in check_export.jl):
  * `:polynomial` is compared to the FITTED ET stack at 1e-12, on both models.  No tolerance
    in this file may ever be loosened.
=#

using Test
using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
using AtomsBase
using Unitful

include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

const MS3_ELEMENTS = (:Ti, :Al, :V)
const MS3_NZ = 3
const MS3_PAIR_MAXN = 8
# One cutoff drives the model, the neighbour lists and every check_export call.
const MS3_RCUT = 6.1

"""
    ms3_model(kind) -> (model, ps, st, rcut)

`kind = :asym`    -- per-pair many-body cutoffs `4.6 + 0.2i + 0.3j` (max = `MS3_RCUT`).
`kind = :uniform` -- one cutoff `MS3_RCUT`, per-symmetric-pair `r0` instead.

Both have a live pair term.  The pair *envelope* cutoff must be the same for every pair in
both variants: `ETModels._convert_pair_envelope` asserts `env == env1` over the whole
NZ x NZ envelope matrix, so only the pair `r0` may vary.  `rcut` is returned rather than
hard-coded at the call sites, and is asserted to equal the cutoff `convert2et_full` picks.
"""
function ms3_model(kind::Symbol)
    d = M._default_rin0cuts(MS3_ELEMENTS)
    mb = if kind == :asym
        SMatrix{3,3}([(rin = 0.0, r0 = d[i, j].r0, rcut = 4.6 + 0.2i + 0.3j) for i in 1:3, j in 1:3])
    elseif kind == :uniform
        SMatrix{3,3}([(rin = 0.0, r0 = 2.3 + 0.13 * min(i, j) + 0.17 * max(i, j), rcut = MS3_RCUT)
                      for i in 1:3, j in 1:3])
    else
        error("ms3_model: unknown kind $kind")
    end
    rcut = maximum(c.rcut for c in mb)
    pair_r0(i, j) = 2.3 + 0.13 * min(i, j) + 0.17 * max(i, j)
    pair_rin0cuts = SMatrix{3,3}([(rin = 0.0, r0 = pair_r0(i, j), rcut = rcut) for i in 1:3, j in 1:3])
    pair_basis = M.ace_learnable_Rnlrzz(; elements = MS3_ELEMENTS, level = M.TotalDegree(),
                                          max_level = MS3_PAIR_MAXN, maxl = 0,
                                          maxn = MS3_PAIR_MAXN, rin0cuts = pair_rin0cuts,
                                          transforms = (:agnesi, 1, 4), envelopes = :poly1sr,
                                          Winit = :glorot_normal)
    model = M.ace_model(; elements = MS3_ELEMENTS, order = 3, Ytype = :solid,
                          level = M.TotalDegree(), max_level = 6,
                          rin0cuts = mb, pair_basis = pair_basis,
                          init_WB = :glorot_normal, init_Wpair = :glorot_normal,
                          init_Wradial = :glorot_normal,
                          E0s = Dict(:Ti => -1.1, :Al => -2.2, :V => -3.3))
    ps, st = Lux.setup(MersenneTwister(7), model)
    return model, ps, st, rcut
end

"Rattled BCC supercells, 54 atoms each, species cycled Ti/Al/V so every ordered pair occurs."
function ms3_configs(n)
    a = 3.3
    box = [SVector(3a, 0.0, 0.0), SVector(0.0, 3a, 0.0), SVector(0.0, 0.0, 3a)]
    out = []
    for k in 1:n
        rng = MersenneTwister(100 + k)
        pos = SVector{3,Float64}[]
        spec = Symbol[]
        idx = 0
        for ix in 0:2, iy in 0:2, iz in 0:2, b in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
            idx += 1
            p = a .* (SVector(ix, iy, iz) .+ SVector(b))
            push!(pos, p + 0.15 * randn(rng, SVector{3,Float64}))
            push!(spec, MS3_ELEMENTS[mod1(idx + k, 3)])
        end
        push!(out, AtomsBase.periodic_system([e => x * u"Å" for (e, x) in zip(spec, pos)],
                                             box .* u"Å"))
    end
    return out
end

"The six per-symmetric-pair Agnesi tuples of the many-body / pair branches of a stack."
ms3_mb_params(stacked) = stacked.calcs[end].model.rembed.layer.trans.refstate.params
ms3_pair_params(stacked) = stacked.calcs[2].model.rembed.layer.rbasis.trans.refstate.params
# UPSTREAM's symmetric index, not a local re-derivation: this is the function
# ETModels._convert_agnesi actually selects with (via ET.catcat2idx_sym), so comparing the
# exported tables against it checks the generator against the model rather than against a
# second copy of the generator's own formula.
ms3_symidx(i, j, NZ) = ET.symidx(i, j, NZ)

@testset "Multi-Species ETACE Export" verbose = true begin

    model_a, ps_a, st_a, rcut_a = ms3_model(:asym)
    model_u, ps_u, st_u, rcut_u = ms3_model(:uniform)
    stacked_a = ETM.convert2et_full(model_a, ps_a, st_a)
    stacked_u = ETM.convert2et_full(model_u, ps_u, st_u)
    held = ms3_configs(3)
    build = mkpath(joinpath(@__DIR__, "build"))
    f_poly = joinpath(build, "ms3_poly.jl")
    f_poly_u = joinpath(build, "ms3_poly_uniform.jl")

    @testset "the NZ=3 models discriminate the two pair conventions" begin
        for (tag, stacked, rcut) in (("asym", stacked_a, rcut_a), ("uniform", stacked_u, rcut_u))
            @test length(stacked.calcs) == 3
            @test [string(nameof(typeof(c.model))) for c in stacked.calcs] ==
                  ["ETOneBody", "ETPairModel", "ETACE"]
            # ONE cutoff: the model's own maximum, what convert2et_full picked, and what
            # every neighbour list and check_export call below uses.
            @test rcut == MS3_RCUT
            @test stacked.calcs[end].rcut == rcut
            @test stacked.calcs[2].rcut == rcut

            mb_par = ms3_mb_params(stacked)
            pr_par = ms3_pair_params(stacked)
            nsym = (MS3_NZ * (MS3_NZ + 1)) ÷ 2
            # transform parameters are stored per SYMMETRIC pair ...
            @test length(mb_par) == nsym
            @test length(pr_par) == nsym
            # ... and are mutually distinct, which is the ONLY thing that makes a wrong
            # ordered -> symmetric expansion observable (the Cantor fixture has six
            # identical tuples, so no permutation of them could ever be detected).
            @test length(unique(mb_par)) == nsym
            @test length(unique(pr_par)) == nsym

            # Weights are per ORDERED pair and are genuinely asymmetric under (i,j)->(j,i).
            # (The transform parameters cannot be: one tuple is shared by both orderings.)
            W = stacked.calcs[end].ps.rembed.post.W
            Wp = stacked.calcs[2].ps.rembed.rbasis.post.W
            @test size(W, 3) == MS3_NZ^2
            @test size(Wp, 3) == MS3_NZ^2
            @test all(W[:, :, (i - 1) * MS3_NZ + j] != W[:, :, (j - 1) * MS3_NZ + i]
                      for i in 1:MS3_NZ, j in 1:MS3_NZ if i != j)
            @test all(Wp[:, :, (i - 1) * MS3_NZ + j] != Wp[:, :, (j - 1) * MS3_NZ + i]
                      for i in 1:MS3_NZ, j in 1:MS3_NZ if i != j)
            @info "NZ=3 $tag model: $nsym distinct symmetric transform tuples, asymmetric ordered weights"
        end
    end

    @testset ":polynomial (per-pair cutoffs) vs the fitted ET stack, 1e-12" begin
        Base.invokelatest(export_ace_model, stacked_a, f_poly; radial_basis = :polynomial)
        @test isfile(f_poly)
        # check_export_report measures without asserting, so the three @test lines below ARE
        # the gate (check_export's internal @assert would otherwise make them unfailable).
        dE, dF, dV = Base.invokelatest(check_export_report, f_poly, stacked_a, held, rcut_a;
                                       label = "NZ=3 asym :polynomial vs FITTED stack")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12
    end

    @testset ":polynomial (uniform cutoff) vs the fitted ET stack, 1e-12" begin
        # The `:uniform` model used to carry the `:hermite_spline` gate, because it was the
        # only one of the two that could be splinified at all.  With the spline export gone
        # it is gated the same way the `:asym` model is.  It is not redundant: its per-pair
        # Agnesi tuples differ in `r0` rather than in `rcut`, so a wrong ordered -> symmetric
        # expansion shows up in a different table.
        Base.invokelatest(export_ace_model, stacked_u, f_poly_u; radial_basis = :polynomial)
        @test isfile(f_poly_u)
        dE, dF, dV = Base.invokelatest(check_export_report, f_poly_u, stacked_u, held, rcut_u;
                                       label = "NZ=3 uniform :polynomial vs FITTED stack")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12
    end

    @testset "a splinified model is refused, whatever the keyword" begin
        # WAS "upstream: splinified evaluation is impossible with per-pair cutoffs", which
        # locked the `EquivariantTensors._spl_grid` BoundsError because a Hermite export of
        # the `:asym` model had no evaluable reference.  There is no Hermite export now, so
        # that lock guarded nothing about this repository's behaviour and went; the upstream
        # bug is recorded in export/bench/FINDINGS_parity.md §5 instead.
        #
        # What replaces it is the thing that IS this repository's behaviour: `splinify` still
        # exists and still runs, so a user can still hold a splinified model, and the
        # exporter must refuse it loudly and say what to do -- in the DEFAULT mode, which is
        # how the refusal will actually be met.
        spl_u = ETM.splinify(stacked_u.calcs[end].model, stacked_u.calcs[end].ps,
                             stacked_u.calcs[end].st; Nspl = 50)
        p, st_s = LuxCore.setup(MersenneTwister(1), spl_u)
        p.readout.W .= stacked_u.calcs[end].ps.readout.W
        spl_stack = ETM.StackedCalculator((stacked_u.calcs[1], stacked_u.calcs[2],
                                           ETM.ETACEPotential(spl_u, p, st_s, rcut_u)))

        function _refusal(calc, tag, kw...)
            f = joinpath(build, "ms3_refused_$tag.jl")
            isfile(f) && rm(f)
            err = try
                Base.invokelatest(export_ace_model, calc, f; kw...)
                nothing
            catch e
                e
            end
            return (; err, msg = err === nothing ? "" : sprint(showerror, err), f)
        end

        # (a) the DEFAULT mode -- no radial_basis keyword at all.
        r = _refusal(spl_stack, "default")
        @test r.err isa ErrorException
        @test occursin("already been splinified", r.msg)
        @test occursin("FIT-ON-SPLINES DEPLOYMENT ROUTE IS THEREFORE RETIRED", r.msg)
        @test occursin("BEFORE splinify()", r.msg)          # the workflow that works
        @test isfile(r.f) == false                          # refused before any file opened

        # (b) :polynomial explicitly -- the same refusal.
        r = _refusal(spl_stack, "poly", :radial_basis => :polynomial)
        @test r.err isa ErrorException
        @test occursin("already been splinified", r.msg)
        @test isfile(r.f) == false

        # (c) the removed keyword, on an UNSPLINIFIED model.  It used to be demoted to
        # :polynomial behind a @warn; now the mode does not exist, so it raises and names
        # the remedy rather than silently exporting a different mode from the one asked for.
        r = _refusal(stacked_u, "hermite_kw", :radial_basis => :hermite_spline)
        @test r.err isa ErrorException
        @test occursin("REMOVED", r.msg)
        @test occursin("radial_basis=:polynomial", r.msg)
        @test isfile(r.f) == false

        # The reachable remedy actually works: the SAME model, exported as it was before
        # splinify() was applied, in the default mode.  (That export is gated at 1e-12
        # against the fitted stack in the :polynomial testset above.)
        f_ok = joinpath(build, "ms3_uniform_poly_ok.jl")
        Base.invokelatest(export_ace_model, stacked_u, f_ok)
        @test isfile(f_ok)
    end

    @testset "one ordered pair index keys every per-pair table" begin
        NZ = MS3_NZ

        # --- :polynomial export (the :asym model) ----------------------------------------
        ex = Base.invokelatest(load_exported, f_poly)
        @test ex.NZ == NZ
        @test all(Base.invokelatest(ex.pair_idx, i, j) == (i - 1) * NZ + j
                  for i in 1:NZ, j in 1:NZ)
        # the old symmetric helper is gone from both the module and the emitted source
        @test isdefined(ex, :zz2pair_sym) == false
        @test occursin("zz2pair_sym", read(f_poly, String)) == false

        mb_par = ms3_mb_params(stacked_a)
        pr_par = ms3_pair_params(stacked_a)
        W = stacked_a.calcs[end].ps.rembed.post.W
        @test length(ex.TRANSFORM_PARAMS) == NZ^2
        @test length(ex.PAIR_C) == NZ^2
        @test length(ex.PAIR_TRANSFORM_PARAMS) == NZ^2

        # Since Task 5 the radial mixing is emitted from W's SPARSITY STRUCTURE rather than
        # as one dense SMatrix{N_RNL, N_POLYS} per pair: `RBASIS_ROWS_k` names the rows the
        # pair needs and `RBASIS_W_k` (dense W, which is what :glorot_normal gives) or
        # `RBASIS_SEL_k` (one-hot W) carries the mixing for exactly those rows.  The check
        # below is the same check as before -- that table k holds the ORDERED pair k's
        # weights, not the symmetric pair's -- read off the new shape.
        @test ex.RBASIS_ONEHOT == false      # :glorot_normal must NOT be mistaken for one-hot
        # RNL_USED must BE the set the A basis reads, not merely a set the generator says it
        # is: recompute it here from the emitted ABASIS_SPEC, which is the table the runtime
        # forward and backward passes actually index with.
        @test collect(ex.RNL_USED) == sort(unique(first.(ex.ABASIS_SPEC)))
        for i in 1:NZ, j in 1:NZ
            k = (i - 1) * NZ + j
            p = mb_par[ms3_symidx(i, j, NZ)]
            q = ex.TRANSFORM_PARAMS[k]
            @test (q.rin, q.req, q.a, q.b0, q.b1) == (p.rin, p.req, p.a, p.b0, p.b1)
            rows = collect(getfield(ex, Symbol("RBASIS_ROWS_$k")))
            @test issubset(rows, collect(ex.RNL_USED))
            @test getfield(ex, Symbol("RBASIS_W_$k")) == W[rows, :, k]
            # every row the generator dropped is either unread by the A basis or identically
            # zero for this pair -- i.e. the pruning removed nothing that could be observed
            @test all(t in rows || !(t in ex.RNL_USED) || all(==(0.0), W[t, :, k])
                      for t in 1:size(W, 1))
            pp = pr_par[ms3_symidx(i, j, NZ)]
            qq = ex.PAIR_TRANSFORM_PARAMS[k]
            @test (qq.rin, qq.req, qq.a, qq.b0, qq.b1) == (pp.rin, pp.req, pp.a, pp.b0, pp.b1)
        end
        # The whole point of the NZ=3 model: transposed pairs must be observably different in
        # the emitted tables, which is what makes a symmetric index into an ordered table
        # detectable.  (Before Task 5 this was `RBASIS_W[k] != RBASIS_W[k']`.)
        @test all(getfield(ex, Symbol("RBASIS_W_$((i - 1) * NZ + j)")) !=
                  getfield(ex, Symbol("RBASIS_W_$((j - 1) * NZ + i)"))
                  for i in 1:NZ, j in 1:NZ if i != j)

        # --- :polynomial export of the :uniform model ------------------------------------
        # This half used to check the Hermite knot tables (PAIR_k_F / PAIR_k_ROWS /
        # PAIR_k_REQ / PAIR_k_B0) of the splinified :uniform model.  Those constants are not
        # emitted by any mode now.  The claim it was making -- one ordered pair index keys
        # every per-pair table, and the symmetric Agnesi storage is expanded into it
        # correctly -- is re-made against the :uniform model's :polynomial export, whose
        # per-pair tables are TRANSFORM_PARAMS and RBASIS_W.
        exu = Base.invokelatest(load_exported, f_poly_u)
        @test exu.NZ == NZ
        @test all(Base.invokelatest(exu.pair_idx, i, j) == (i - 1) * NZ + j
                  for i in 1:NZ, j in 1:NZ)
        @test isdefined(exu, :zz2pair_sym) == false
        @test occursin("zz2pair_sym", read(f_poly_u, String)) == false
        @test collect(exu.RNL_USED) == sort(unique(first.(exu.ABASIS_SPEC)))

        mb_par_u = ms3_mb_params(stacked_u)
        for i in 1:NZ, j in 1:NZ
            k = (i - 1) * NZ + j
            qq = exu.TRANSFORM_PARAMS[k]
            pp = mb_par_u[ms3_symidx(i, j, NZ)]
            @test (qq.rin, qq.req, qq.a, qq.b0, qq.b1) == (pp.rin, pp.req, pp.a, pp.b0, pp.b1)
        end
        # the :uniform model's six symmetric tuples differ in r0, so a symmetric index into
        # an ordered table is observable here too
        @test all(getfield(exu, Symbol("RBASIS_W_$((i - 1) * NZ + j)")) !=
                  getfield(exu, Symbol("RBASIS_W_$((j - 1) * NZ + i)"))
                  for i in 1:NZ, j in 1:NZ if i != j)
    end

    # Task 6 removed the neighbour cap: there is no MAX_NEIGHBORS constant any more, and the
    # scratch lives in a Workspace every one of whose buffers is sized from the MODEL (N_A,
    # N_AA, N_BASIS) rather than from the neighbour count -- nothing is resized per site, which
    # is why there is no cap left to hit.  What is asserted here is therefore stronger than
    # before: a 300-neighbour site must EVALUATE, and the emitted source must not carry the
    # constant or the old global work arrays.
    #
    # It is still not proof of the absence of a silent cap -- a build that quietly truncated
    # the neighbour list would return a plausible finite number and pass.  The gate that
    # actually catches truncation is the 1e-12 comparison against the Julia calculator on the
    # real held-out configurations (check_export), whose Cantor sites carry ~200 neighbours.
    @testset "a 300-neighbour site evaluates; the cap and the global scratch are gone" begin
        ex = Base.invokelatest(load_exported, f_poly)
        src = read(f_poly, String)
        @test !occursin("MAX_NEIGHBORS", src)
        @test !occursin("WORK_", src)
        @test isdefined(ex, :MAX_NEIGHBORS) == false

        n = 300
        Rs = [SVector(2.0 + 0.001k, 0.1, -0.05) for k in 1:n]
        Zs = fill(22, n)
        E = Base.invokelatest(ex.site_energy, Rs, Zs, 22)
        @test isfinite(E)
        Ef, Ff, Vf = Base.invokelatest(ex.site_energy_forces_virial, Rs, Zs, 22)
        @test isfinite(Ef) && length(Ff) == n && all(isfinite, Vf)
        # the same site through an explicitly supplied workspace is bitwise identical
        ws = Base.invokelatest(ex.new_workspace)
        F2 = Vector{SVector{3, Float64}}(undef, n)
        E2, V2 = Base.invokelatest(ex.site_energy_forces_virial!, ws, Rs, Zs, 22, F2)
        @test E2 === Ef && F2 == Ff && V2 == Vf
    end

end
