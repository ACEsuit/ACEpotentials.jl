#=
Multi-species export tests -- NZ = 3, both radial modes, gated at 1e-12.

WHY THESE MODELS.  Every per-pair table in a generated model is addressed by a species-pair
index, and there are two candidate conventions:

    ordered    k = (iz - 1) * NZ + jz                  (NZ^2 entries, centre species first)
    symmetric  k = symidx(min, max, NZ)                (NZ(NZ+1)/2 entries)

The *model* uses both: `ET.SelectLinL` weights (many-body `RBASIS_W`, the pair `PAIR_C`) and
the splinified knot tables are per ORDERED pair, while the Agnesi transform parameters are
stored per SYMMETRIC pair (`ETModels._convert_agnesi` loops `for i = 1:NZ, j = i:NZ` and
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
              cutoffs.  Works in `:polynomial` mode.  In `:hermite_spline` mode the
              SPLINIFIED ET model itself cannot be evaluated: any edge with
              `rcut[i,j] <= r <= max(rcut)` transforms to exactly `y = 1`, and
              `EquivariantTensors`' `_spl_grid` then indexes knot `NX + 1`
              (`transsplines.jl:200-207`, `BoundsError: ... at index [51, 5]`).  There is
              therefore no reference to gate a Hermite export against.  That is asserted
              below so the day upstream fixes it, this file fails and coverage is extended.
  `:uniform`  one cutoff for every pair, with a per-symmetric-pair `r0` instead.  Equally
              discriminating for the index convention, and evaluable in both modes, so it
              carries the `:hermite_spline` gate.

REFERENCES AND TOLERANCES (metric definitions are in check_export.jl):
  * `:polynomial`     is compared to the FITTED ET stack at 1e-12.
  * `:hermite_spline` is compared to the SPLINIFIED stack at 1e-12; its error against the
    fitted stack is REPORTED and never asserted -- that is model (splinification) error, not
    export error.  No tolerance in this file may ever be loosened.
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

"""
    ms3_spline_stack(stacked; Nspl) -> StackedCalculator

`(ETOneBody, ETPairModel, splinified ETACE)`.  The pair term is carried over UNSPLINIFIED --
`splinify` only touches the many-body radial basis, and the exporter emits the pair term in
both radial modes, so a pair-less reference would re-measure the defect Task 1 removed.
Built exactly like `cantor_spline_stack` in fixtures/cantor_fixture.jl.
"""
function ms3_spline_stack(stacked; Nspl::Integer)
    onebody, pair, ace = stacked.calcs
    m = ETM.splinify(ace.model, ace.ps, ace.st; Nspl = Nspl)
    p, s = LuxCore.setup(MersenneTwister(1), m)
    p.readout.W .= ace.ps.readout.W
    return ETM.StackedCalculator((onebody, pair, ETM.ETACEPotential(m, p, s, ace.rcut)))
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
    f_herm = joinpath(build, "ms3_hermite50.jl")

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

    @testset ":hermite_spline (uniform cutoff) vs the SPLINIFIED stack, 1e-12" begin
        spl = ms3_spline_stack(stacked_u; Nspl = 50)
        # the SPLINIFIED stack is what gets exported -- export_ace_model will not emit
        # Hermite tables for an unsplinified model (it warns and demotes to :polynomial),
        # which would turn this gate into a comparison of two different models
        Base.invokelatest(export_ace_model, spl, f_herm; radial_basis = :hermite_spline)
        @test isfile(f_herm)
        # export_ace_model auto-detects splinification; the file must really be the spline one
        @test occursin("HERMITE CUBIC SPLINE RADIAL BASIS", read(f_herm, String))

        # as above: check_export_report measures, the @test lines gate.
        dE, dF, dV = Base.invokelatest(check_export_report, f_herm, spl, held, rcut_u;
                                       label = "NZ=3 uniform :hermite_spline(Nspl=50) vs SPLINIFIED stack")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12

        # Informational ONLY.  This is splinification (model) error, never export error, and
        # it must never be compared against any tolerance.
        fit = Base.invokelatest(check_export_report, f_herm, stacked_u, held, rcut_u;
                                label = "NZ=3 uniform :hermite_spline(Nspl=50) vs FITTED stack [informational]")
        @info "Hermite error against the FITTED model -- reported, never gated" dE_atom = fit[1] dF = fit[2] dV_atom = fit[3]
    end

    @testset "upstream: splinified evaluation is impossible with per-pair cutoffs" begin
        # EquivariantTensors `_spl_grid` (transsplines.jl:200-207) clamps y to [x0, x1] and
        # then takes knots `il+1, il+2`; at y == x1 that is knot NX+1.  Any edge beyond a
        # pair's own cutoff but inside the neighbour cutoff transforms to exactly y = 1, so
        # the splinified ET model throws before the exported model can be compared to it.
        # The EXPORTED code clamps the segment index and evaluates fine -- it is the
        # reference, not the export, that is missing.  When this test starts failing,
        # upstream has been fixed: move the Hermite gate above onto the :asym model.
        spl_a = ms3_spline_stack(stacked_a; Nspl = 50)
        @test_throws BoundsError AtomsCalculators.potential_energy(held[1], spl_a)

        # ... and because that reference cannot be evaluated, the exporter must REFUSE to emit
        # such a model rather than produce an artefact no gate can ever check.
        #
        # BOTH keywords must be refused for a SPLINIFIED per-pair-cutoff model, but for two
        # DIFFERENT reasons, and the tests below pin each one:
        #
        #   :hermite_spline -> the per-pair-cutoff refusal (this model cannot be verified);
        #   :polynomial     -> the splinification refusal (Task 3 / K4: a splinified model has
        #                      no polynomial recurrence left to emit, and this used to be
        #                      silently promoted to :hermite_spline behind a @warn).
        #
        # Either way no file is produced, and neither message may offer :polynomial as the
        # remedy for THIS model -- asserted explicitly below.
        function _refusal(calc, tag, kw)
            f = joinpath(build, "ms3_hermite_refused_$tag.jl")
            isfile(f) && rm(f)
            err = try
                Base.invokelatest(export_ace_model, calc, f; radial_basis = kw)
                nothing
            catch e
                e
            end
            return (; err, msg = err === nothing ? "" : sprint(showerror, err), f)
        end

        # (a) :hermite_spline -- refused because the pairs do not share one cutoff.
        r = _refusal(spl_a, "hermite", :hermite_spline)
        @test r.err isa ErrorException
        @test occursin("radial_basis=:hermite_spline requires every species pair", r.msg)
        @test occursin("pair 1 = (iz=1, jz=1)", r.msg)   # 5.1 Å, short of RCUT_MAX = 6.1
        @test isfile(r.f) == false                        # refused before any file opened
        # the advice must be reachable for a caller who is already splinified
        @test occursin("passing radial_basis=:polynomial is NOT one of them", r.msg)
        @test occursin("BEFORE splinify()", r.msg)

        # (b) :polynomial -- refused because the model is splinified.  This is the K4 change:
        # it used to be PROMOTED to :hermite_spline behind a @warn, so a caller who asked for
        # the exact mode (or simply took the default) received the approximate one.
        r = _refusal(spl_a, "splinified_poly", :polynomial)
        @test r.err isa ErrorException
        @test occursin("radial_basis=:polynomial cannot export this model", r.msg)
        @test occursin("already been splinified", r.msg)
        @test occursin(":polynomial is the DEFAULT", r.msg)          # fires with no keyword too
        @test occursin("BEFORE splinify()", r.msg)                   # remedy 1
        @test occursin("radial_basis=:hermite_spline explicitly", r.msg)  # remedy 2, the opt-in
        @test isfile(r.f) == false                                   # no partial file

        # and the default keyword -- no radial_basis at all -- must hit the same refusal.
        f_def = joinpath(build, "ms3_default_refused.jl")
        isfile(f_def) && rm(f_def)
        err_def = try
            Base.invokelatest(export_ace_model, spl_a, f_def)
            nothing
        catch e
            e
        end
        @test err_def isa ErrorException
        @test occursin("radial_basis=:polynomial cannot export this model",
                       err_def === nothing ? "" : sprint(showerror, err_def))
        @test isfile(f_def) == false

        # The reachable remedy actually works: the SAME model, exported as it was before
        # splinify() was applied, in the default mode.  (That export is gated at 1e-12
        # against the fitted stack in the :polynomial testset above.)
        f_ok = joinpath(build, "ms3_asym_poly_ok.jl")
        Base.invokelatest(export_ace_model, stacked_a, f_ok; radial_basis = :polynomial)
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

        # --- :hermite_spline export (the :uniform model) ---------------------------------
        exh = Base.invokelatest(load_exported, f_herm)
        @test all(Base.invokelatest(exh.pair_idx, i, j) == (i - 1) * NZ + j
                  for i in 1:NZ, j in 1:NZ)
        @test isdefined(exh, :zz2pair_sym) == false
        @test occursin("zz2pair_sym", read(f_herm, String)) == false

        @test collect(exh.RNL_USED) == sort(unique(first.(exh.ABASIS_SPEC)))

        mb_par_u = ms3_mb_params(stacked_u)
        F = ms3_spline_stack(stacked_u; Nspl = 50).calcs[end].st.rembed.params.F
        @test size(F, 2) == NZ^2          # the knot tables are per ORDERED pair
        for i in 1:NZ, j in 1:NZ
            k = (i - 1) * NZ + j
            @test isdefined(exh, Symbol("PAIR_$(k)_F"))
            Fk = getfield(exh, Symbol("PAIR_$(k)_F"))
            @test length(Fk) == size(F, 1)
            # Since Task 5 the knot tables carry only PAIR_k_ROWS, the rows the A basis reads
            # AND this pair populates; compare against exactly those rows of the model's F,
            # and check separately that every row left out really is unread or zero.
            rows = collect(getfield(exh, Symbol("PAIR_$(k)_ROWS")))
            @test issubset(rows, collect(exh.RNL_USED))
            @test all(collect(Fk[t]) ≈ collect(F[t, k])[rows] for t in 1:size(F, 1))
            @test all(t in rows || !(t in exh.RNL_USED) ||
                      all(collect(F[s, k])[t] == 0.0 for s in 1:size(F, 1))
                      for t in 1:exh.N_RNL)
            p = mb_par_u[ms3_symidx(i, j, NZ)]
            @test getfield(exh, Symbol("PAIR_$(k)_REQ")) == p.req
            @test getfield(exh, Symbol("PAIR_$(k)_B0")) == p.b0
        end
    end

    # Task 6 removed the neighbour cap: there is no MAX_NEIGHBORS constant any more and the
    # scratch arrays live in a per-call Workspace that is resize!d to the site.  What is
    # asserted here is therefore stronger than before: a 300-neighbour site must EVALUATE,
    # and the emitted source must not carry the constant or the old global work arrays.
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
