#!/usr/bin/env julia
#=
diag_dA_conditioning.jl [cantor|tial ...]  -- how accurate is `∂A = ∂E/∂A`, really?

    julia --project=export export/bench/diag_dA_conditioning.jl cantor tial

THIS IS THE ARTEFACT BEHIND the BigFloat / κ table in `export/bench/README.md` and section 4b
of the Task 7 report, and behind the note now carried at every 1e-12 force gate
(`export/test/check_export.jl`, `export/bench/verify_bench_models.jl`,
`export/test/test_generator_parity.jl`).

WHAT IT ANSWERS.  The plan gates the exported forces against `ETACEPotential` at 1e-12
absolute. That gate is only meaningful if `∂A` — the one intermediate a change to the tensor
step can move — is itself determined to better than 1e-12 in Float64. On the TiAl order-4
model it is NOT, and this script is how that was established rather than argued:

  * `flat_route`  reproduces the SHIPPED (`aa_products = :flat`) tensor step: flat AA
    products, `∂AA = A2Bmapᵀ·WB`, flat pullback. It is generic in the element type.
  * `dag_route`   reproduces `aa_products = :dag`: the DAG forward, `CTILDE`-seeded backward.
  * Both are run in Float64 AND in BigFloat on the SAME real `A` vectors, taken from an
    exported model's own `_embed_val!` on held-out configurations.
  * The two BigFloat runs are required to agree to 1e-40 relative. THEY MUST, and that
    requirement is the check that the DAG is mathematically exact rather than merely close;
    if it fires, something is wrong with the DAG, not with the arithmetic.
  * `κ = Σ|contributions to ∂A[i]| / |∂A[i]|` is the cancellation condition number, and
    `κ·eps·|∂A|` is the floor no Float64 route can beat. Each route's Float64 error against
    the BigFloat value is reported as a multiple of that floor.

A NOTE ON GETTING THIS WRONG, because the first run of it did: `CTILDE` must be recomputed
IN BigFloat. Promoting the Float64 `CTILDE` makes the two "exact" routes use different
coefficients, and they then disagree at ~1e-16 and the 1e-40 assertion fires for a reason that
has nothing to do with the DAG.

Reads `bench_parity/<model>_poly_b3_model.jl` only to obtain realistic `A` vectors; any export
of the same model would do.
=#
using Printf, LinearAlgebra, SparseArrays

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "export", "test", "check_export.jl"))          # + cantor fixture
include(joinpath(REPO, "export", "test", "fixtures", "tial_fixture.jl"))
include(joinpath(REPO, "export", "src", "export_ace_model.jl"))       # SymmProdDAG et al.

const MODELS = isempty(ARGS) ? ["cantor", "tial"] : filter(a -> !startswith(a, "--"), ARGS)
const NSITES = 20
const NCONF = 4

"The SHIPPED tensor step, generic in T.  `want_kappa` also returns Σ|contributions| per ∂A[i]."
function flat_route(A::Vector{T}, aab, A2B, WB; want_kappa = false) where {T}
    I, J, V = findnz(A2B)
    ∂AA = zeros(T, size(A2B, 2))
    for k in eachindex(I); ∂AA[J[k]] += T(V[k]) * WB[I[k]]; end
    ∂A = zeros(T, length(A)); absA = zeros(T, length(A))
    for (ord, s) in enumerate(aab.specs)
        r0 = aab.ranges[ord].start
        for (i, ϕ) in enumerate(s)
            g = ∂AA[r0 - 1 + i]
            for t in 1:ord
                p = g
                for u in 1:ord; u == t || (p *= A[ϕ[u]]); end
                ∂A[ϕ[t]] += p
                want_kappa && (absA[ϕ[t]] += abs(p))
            end
        end
    end
    return ∂A, absA
end

"`aa_products = :dag`: DAG forward, CTILDE-seeded backward."
function dag_route(A::Vector{T}, dag, ct::Vector{T}) where {T}
    AAd = evaluate_dag(dag, A)
    return pullback_dag(dag, ct, AAd)
end

"CTILDE computed natively in T (see the header: promoting a Float64 CTILDE invalidates this)."
function ctilde_in(::Type{T}, dag, A2B, WB::Vector{T}) where {T}
    I, J, V = findnz(A2B)
    ctf = zeros(T, size(A2B, 2))
    for k in eachindex(I); ctf[J[k]] += T(V[k]) * WB[I[k]]; end
    ct = zeros(T, length(dag.nodes))
    for (k, n) in enumerate(dag.projection); ct[n] += ctf[k]; end
    return ct
end

function run(model)
    fx = model == "cantor" ? load_cantor_fixture() : load_tial_fixture()
    _, _, ace = model == "cantor" ? cantor_substacks(fx) : tial_substacks(fx)
    tensor = ace.model.basis; aab = tensor.aabasis; A2B = tensor.A2Bmaps[1]
    dag = SymmProdDAG(aa_flat_spec(aab))
    ex = load_exported(joinpath(REPO, "bench_parity", "$(model)_poly_b3_model.jl"))
    zl = [Int(z.atomic_number) for z in ace.model.rembed.layer.trans.refstate.zlist]
    allsites = [s for c in 1:NCONF for s in Base.invokelatest(site_sets, fx.held[c], fx.rcut)]

    println("\n", uppercase(model), "   N_A=", length(tensor.abasis),
            "  N_AA=", length(aab), "  N_DAG=", length(dag.nodes))
    @printf("  %-10s %10s %8s %12s   %-22s %-22s\n",
            "species", "max|∂A|", "kappa", "floor", ":flat err (x floor)", ":dag err (x floor)")
    for iz in eachindex(zl)
        WB64 = ace.ps.readout.W[1, :, iz]
        ct64 = ctilde_in(Float64, dag, A2B, WB64)
        WBbf = BigFloat.(WB64)
        ctbf = ctilde_in(BigFloat, dag, A2B, WBbf)
        wf = wd = sc = fl = 0.0; kmax = 0.0; ns = 0
        for (Rs, Zs, Z0, _) in allsites
            findfirst(==(Int(Z0)), zl) == iz || continue
            ns += 1; ns > NSITES && break
            ws = Base.invokelatest(ex.new_workspace)
            Base.invokelatest(ex._embed_val!, ws, Rs, Zs, iz)
            A64 = copy(ws.A); Abf = BigFloat.(A64)
            dA_flat, _ = flat_route(A64, aab, A2B, WB64)
            dA_dag = dag_route(A64, dag, ct64)
            dA_ex, absA = flat_route(Abf, aab, A2B, WBbf; want_kappa = true)
            dA_ex2 = dag_route(Abf, dag, ctbf)
            rel = Float64(maximum(abs.(dA_ex .- dA_ex2)) / maximum(abs.(dA_ex)))
            rel < 1e-40 || error("""
                the two BigFloat routes disagree by $rel relative on $model species $iz.
                They compute the same mathematical quantity, so this is a DAG defect (or a
                Float64 CTILDE promoted into BigFloat -- see the header), not roundoff.""")
            wf = max(wf, maximum(abs.(Float64.(dA_flat) .- Float64.(dA_ex))))
            wd = max(wd, maximum(abs.(dA_dag .- Float64.(dA_ex))))
            sc = max(sc, Float64(maximum(abs.(dA_ex))))
            kmax = max(kmax, Float64(maximum(absA) / maximum(abs.(dA_ex))))
            fl = max(fl, Float64(maximum(absA)) * eps())
        end
        @printf("  Z=%-8d %10.4g %8.3g %12.4g   %10.4g (%5.2fx)      %10.4g (%5.2fx)\n",
                zl[iz], sc, kmax, fl, wf, wf / fl, wd, wd / fl)
    end
end

println("diag_dA_conditioning.jl -- ", NSITES, " sites/species from ", NCONF, " held-out configurations")
for m in MODELS
    run(m)
end
