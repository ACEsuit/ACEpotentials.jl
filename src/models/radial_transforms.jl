
import ForwardDiff

struct GeneralizedAgnesiTransform{T}
   p::Int
   q::Int 
   a::T
   rin::T 
   r0::T 
end

(t::GeneralizedAgnesiTransform)(r) = evaluate(t, r)

write_dict(T::GeneralizedAgnesiTransform) =
      Dict("__id__" => "ACEpotentials_GeneralizedAgnesiTransform", 
           "r0" => T.r0, "p" => T.p, "q" => T.q, "a" => T.a, "rin" => T.rin)

GeneralizedAgnesiTransform(D::Dict) = 
      GeneralizedAgnesiTransform(D["r0"], D["p"], D["q"], 
                                 D["a"], D["rin"])

read_dict(::Val{:ACEpotentials_GeneralizedAgnesiTransform}, D::Dict) = 
      GeneralizedAgnesiTransform(D)

function evaluate(t::GeneralizedAgnesiTransform{T}, r::Number) where {T} 
   if r <= t.rin 
      return one(promote_type(T, typeof(r)))
   end 
   a, r0, q, p, rin = t.a, t.r0, t.q, t.p, t.rin
   s = (r-t.rin)/(t.r0-t.rin)
   return 1 / (1 + a * s^q / (1 + s^(q-p)))
end

evaluate_d(t::GeneralizedAgnesiTransform, r::Number) = 
     ForwardDiff.derivative(r -> transform(t, r), r)


     
# ---------------------------------------------------------------------------


# tested a wide range of methods. Brent seems robust + fastest 
using Roots: find_zero, ITP, Brent 

"""
Maps the transform `trans` to the standardized interval [-1, 1]
"""
struct NormalizedTransform{T, TT}
   trans::TT
   yin::T 
   ycut::T
   rin::T 
   rcut::T
end

function NormalizedTransform(trans, rin::Number, rcut::Number) 
   yin = trans(rin)
   ycut = trans(rcut) 
   return NormalizedTransform(trans, yin, ycut, rin, rcut)
end


(t::NormalizedTransform)(r) = evaluate(t, r)

function evaluate(t::NormalizedTransform, r::Number) 
   y = t.trans(r)
   𝟙 = one(typeof(y))
   return min(max(-𝟙, -𝟙 + 2 * (y - t.yin) / (t.ycut - t.yin)), 𝟙)
end

# this is the old version from ACE1.jl; a neat idea. We could return to it. 
# but it could be better to integrate this into the inner transform. 
# return 1 - (y - t.y1) / (t.y0 - t.y1) * (1 - (r/t.rcut)^4)

evaluate_d(t::NormalizedTransform, r::Number) = 
         ForwardDiff.derivative(r -> evaluate(t, r), r)


function inv_transform(t::NormalizedTransform{T}, x::Number) where {T} 
   T1 = promote_type(T, typeof(x))
   𝟙 = one(T1)
   if x <= -𝟙 
      return convert(T1, t.rin)
   elseif x >= 𝟙
      return convert(T1, t.rcut)
   end

   g = r -> evaluate(t, r) - x
   r = find_zero(g, (t.rin, t.rcut), Brent())
   @assert t.rin <= r <= t.rcut
   @assert abs(g(r)) < 1e-12
   return r 
end

write_dict(T::NormalizedTransform) =
      Dict("__id__" => "ACEpotentials_NormalizedTransform",
            "trans" => write_dict(T.trans), 
            "yin" => T.yin, "ycut" => T.ycut, 
            "rin" => T.rin, "rcut" => T.rcut ) 
            
read_dict(::Val{:ACEpotentials_NormalizedTransform}, D::Dict) = 
      NormalizedTransform(read_dict(D["trans"]), 
                          D["yin"], D["ycut"], D["rin"], D["rcut"])

# ---------------------------------------------------------------------------

function test_normalized_transform(t; nx = 1000)
   fails = 0 

   x = range(-1.0, 1.0, length = nx)
   r = [inv_transform(t, xi) for xi in x]
   if !(r[1] ≈ t.rin)
      fails += 1 
      @error("t⁻¹(-1) ≈ rin fails")
   end
   if !(r[end] ≈ t.rcut) 
      fails += 1 
      @error("t⁻¹(1) ≈ rcut fails")
   end
   x2 = [t(ri) for ri in r]
   if !all(abs.(x .- x2) .< 1e-10)
      fails += 1
      @error("t⁻¹ ∘ t ≈ id failed!")
   end
   if !(all(r[2:end] - r[1:end-1] .>= - eps()))
      fails += 1 
      @error("t⁻¹ is not monotonically increasing")
   end

   r = range(0.0, t.rcut, length = nx)
   x = [t(ri) for ri in r]
   if !(x[1] ≈ -1.0)
      fails += 1 
      @error("t(rin) ≈ yin fails")
   end
   if !(x[end] ≈ 1.0)
      fails += 1 
      @error("t(rcut) ≈ ycut fails")
   end
   r2 = [inv_transform(t, xi) for xi in x]
   if !all(abs.(r .- r2) .< 1e-10)
      fails += 1
      @error("t ∘ t⁻¹ ≈ id failed!")
   end
   if !(all(x[2:end] - x[1:end-1] .>= - eps()))
      fails += 1 
      @error("t is not monotonically increasing")
   end

   rr = r[2:end-1]
   dx = evaluate_d.(Ref(t), rr)
   adx = ForwardDiff.derivative.(Ref(r -> t(r)), rr)
   if !all(abs.(dx - adx) .< 1e-10)
      fails += 1 
      @error("transform gradient test failed")
   end

   if fails > 0 
      @info("$fails transform tests fails")
   end 

   # TODO: check that the transform doesn't allocate
   @allocated begin
      x1 = 0.0;
      for r1 in rr
         x1 += evaluate(t, r1)
      end
   end 

   return (fails == 0) 
end




@doc raw"""
`function agnesi_transform:` constructs a generalized agnesi transform. 
```
trans = agnesi_transform(r0, p, q)
```
with `q >= p`. This generates an `AnalyticTransform` object that implements 
```math
   x(r) = \frac{1}{1 + a (r/r_0)^q / (1 + (r/r0)^(q-p))}
```
with default `a` chosen such that $|x'(r)|$ is maximised at $r = r_0$. But `a` may also be specified directly as a keyword argument. 

The transform satisfies 
```math 
   x(r) \sim \frac{1}{1 + a (r/r_0)^p} \quad \text{as} \quad r \to 0 
   \quad \text{and} 
   \quad 
   x(r) \sim \frac{1}{1 + a (r/r_0)^p}  \quad \text{as} r \to \infty.
```

As default parameters we recommend `p = 2, q = 4` and the defaults for `a`.
"""
function agnesi_transform(r0, rcut, p, q;
               rin = zero(r0), 
               a = (-2 * q + p * (-2 + 4 * q)) / (p + p^2 + q + q^2) )
   @assert p > 0
   @assert q > 0
   @assert q >= p      
   @assert a > 0 
   @assert 0 < r0 < rcut 
   return NormalizedTransform( 
                  GeneralizedAgnesiTransform(p, q, a, rin, r0), 
                  rin, rcut )
end

function agnesi_transform(rin0cut::NamedTuple, p, q)
   rin = rin0cut.rin 
   r0 = rin0cut.r0
   rcut = rin0cut.rcut
   return agnesi_transform(r0, rcut, p, q, rin = rin)
end
