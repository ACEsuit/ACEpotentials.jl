#!/usr/bin/env julia
# Check spline data extraction

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify
using StaticArrays
using Random
using Lux
using LuxCore

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
const EXPORT_DIR = dirname(@__DIR__)

include(joinpath(EXPORT_DIR, "src", "splinify.jl"))
include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))

println("\nChecking Spline Data Extraction")
println("="^80)

# Create and splinify model
elements = (:Si,)
rcut = 5.5
rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = 2,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = 8,
    maxl = 2,
    pair_maxn = 8,
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(M._default_rin0cuts(elements)),
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(rng, et_model)

# Copy parameters
for iz in 1:1, jz in 1:1
    et_ps.rembed.post.W[:, :, (iz-1)*1 + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
end

et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
et_ps_splined, et_st_splined = LuxCore.setup(rng, et_model_splined)

# Extract spline data
println("\n[1] Extracting spline data...")
spline_data = extract_hermite_spline_data(et_model_splined, et_ps_splined, et_st_splined, rcut)

pair_idx = 1
hermite = spline_data[pair_idx]

println("   Pair: $(hermite.iz) → $(hermite.jz)")
println("   n_rnl: $(hermite.n_rnl)")
println("   n_knots: $(hermite.n_knots)")
println("   y_min: $(hermite.y_min)")
println("   y_max: $(hermite.y_max)")
println("   F size: $(size(hermite.F))")
println("   G size: $(size(hermite.G))")

# Check values at a few knots
println("\n[2] Spline knot values:")
for ik in [1, 25, 50]
    println("   Knot $ik:")
    println("     F: $(hermite.F[ik, 1:3])...")
    println("     G: $(hermite.G[ik, 1:3])...")
end

# Now let's manually evaluate the Hermite spline at r = 3.0
println("\n[3] Manual Hermite evaluation at r = 3.0...")

# First apply Agnesi transform
r = 3.0
a = hermite.agnesi_params.a
b0 = hermite.agnesi_params.b0
b1 = hermite.agnesi_params.b1
rin = hermite.agnesi_params.rin
req = hermite.agnesi_params.req
pin = hermite.agnesi_params.pin
pcut = hermite.agnesi_params.pcut

s_in = (r - rin)^pin
s_out = pcut > 0 ? (rcut - r)^pcut : 1.0
x = (r - req) / a
y = 2 / (1 + exp(b0 + b1 * x)) - 1

println("   r = $r")
println("   Agnesi params: a=$a, b0=$b0, b1=$b1, rin=$rin, req=$req")
println("   Transformed y = $y")
println("   s_in = $s_in, s_out = $s_out")

# Find segment
h = (hermite.y_max - hermite.y_min) / (hermite.n_knots - 1)
t_raw = (y - hermite.y_min) / h
t_frac, t_floor = modf(t_raw)
il = Int(floor(t_floor)) + 1

println("   Segment: il=$il, t_frac=$t_frac")

# Get knot data
fl = hermite.F[il, :]
fr = hermite.F[il+1, :]
gl = h .* hermite.G[il, :]
gr = h .* hermite.G[il+1, :]

# Hermite cubic
a0 = fl
a1 = gl
a2 = @. -3fl + 3fr - 2gl - gr
a3 = @. 2fl - 2fr + gl + gr
s = @. ((a3 * t_frac + a2) * t_frac + a1) * t_frac + a0

println("   Hermite result (first 5): $(s[1:5])")
println("   Hermite norm: $(norm(s))")

# Apply envelope if present
if hermite.envelope_params !== nothing
    env_a = hermite.envelope_params.a
    env_b0 = hermite.envelope_params.b0
    env_b1 = hermite.envelope_params.b1
    env_rin = hermite.envelope_params.rin
    env_req = hermite.envelope_params.req
    env_pin = hermite.envelope_params.pin
    env_pcut = hermite.envelope_params.pcut

    env_s_in = (r - env_rin)^env_pin
    env_s_out = env_pcut > 0 ? (rcut - r)^env_pcut : 1.0
    env_x = (r - env_req) / env_a
    env_y = 2 / (1 + exp(env_b0 + env_b1 * env_x)) - 1

    println("   Envelope y = $env_y")
    println("   Envelope s_in = $env_s_in, s_out = $env_s_out")

    env = env_s_in * env_s_out
    s_final = env .* s
else
    s_final = s_in * s_out .* s
end

println("   Final Rnl (first 5): $(s_final[1:5])")
println("   Final norm: $(norm(s_final))")

println("\n" * "="^80)
