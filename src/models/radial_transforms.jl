
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
   return min(max(zero(y), (y - t.yin) / (t.ycut - t.yin)), one(y))
end

# this is the old version from ACE1.jl; a neat idea. We could return to it. 
# but it could be better to integrate this into the inner transform. 
# return 1 - (y - t.y1) / (t.y0 - t.y1) * (1 - (r/t.rcut)^4)

evaluate_d(t::NormalizedTransform, r::Number) = 
         ForwardDiff.derivative(r -> evaluate(t, r), r)


function inv_transform(t::NormalizedTransform{T}, x::Number) where {T} 
   T1 = promote_type(T, typeof(x))
   if x <= 0 
      return convert(T1, t.rin)
   elseif x >= 1 
      return convert(T1, t.rcut)
   end

   g = r -> transform(t, r) - x
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
   x = range(0.0, 1.0, length = nx)
   r = [inv_transform(t, xi) for xi in x]
   @assert r[1] == 0.0 
   @assert r[end] == t.rcut
   x2 = [t(ri) for ri in r]
   if !all(abs.(x .- x2) .< 1e-10)
      error("Inverse transform failed!")
   end
   @assert all(r[2:end] - r[1:end-1] .>= 0)

   r = range(0.0, t.rcut, length = nx)
   x = [t(ri) for ri in r]
   @assert x[1] == t.yin
   @assert x[end] == t.ycut
   r2 = [inv_transform(t, xi) for xi in x]
   if !all(abs.(r .- r2) .< 1e-10)
      error("Inverse transform failed!")
   end
   @assert all(x[2:end] .- x[1:end-1] .>= 0)

   return true 
end

# test transform from ACE1 to be merged with the above.
# function test_transform(T, rrange, ntests = 100)

#    rmin, rmax = extrema(rrange)
#    rr = rmin .+ rand(100) * (rmax-rmin)
#    xx = [ transform(T, r) for r in rr ]
#    # check syntactic sugar
#    xx1 = [ T(r) for r in rr ]
#    print_tf(@test xx1 == xx)
#    # check inversion
#    rr1 =  inv_transform.(Ref(T), xx)
#    print_tf(@test rr1 ≈ rr)
#    # check gradient
#    dx = transform_d.(Ref(T), rr)
#    adx = ForwardDiff.derivative.(Ref(r -> transform(T, r)), rr)
#    print_tf(@test dx ≈ adx)

#    # TODO: check that the transform doesn't allocate
#    @allocated begin
#       x = 0.0;
#       for r in rr
#          x += transform(T, r)
#       end
#    end
# end




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
