
abstract type AbstractEnvelope end

struct PolyEnvelope1sR{T}
   rcut::T
   p::Int 
   # ------- 
   meta::Dict{String, Any}
end 


PolyEnvelope1sR(rcut, p) = 
   PolyEnvelope1sR(rcut, p, Dict{String, Any}())

function evaluate(env::PolyEnvelope1sR, r::T, x::T) where T 
   if r >= env.rcut 
      return zero(T)
   end
   p = env.p 
   # return r^(-p) - env.rcut^(-p) - p*(env.rcut^(-p-1))*(r - env.rcut)
   return ( (r/env.rcut)^(-p) - 1.0) * (1 - r / env.rcut)
end

evaluate_d(env::PolyEnvelope1sR, r::T, x::T) where {T} = 
      (ForwardDiff.derivative(x -> evaluate(env, x), r), 
       zero(T),)

# ----------------------------

"""
The pair basis radial envelope implemented in ACE1.jl 
"""
struct ACE1_PolyEnvelope1sR{T}
   rcut::T
   r0::T 
   p::Int 
end 


ACE1_PolyEnvelope1sR(rcut, r0, p) = 
   ACE1_PolyEnvelope1sR(rcut, r0, p, Dict{String, Any}())
   
function evaluate(env::ACE1_PolyEnvelope1sR, r::T, x::T) where T 
   p, r0, rcut = env.p, env.r0, env.rcut   
   if r > rcut; return zero(T); end
   s = r/r0; scut = rcut/r0 
   return s^(-p) - scut^(-p) + p * scut^(-p-1) * (s - scut)
end

evaluate_d(env::ACE1_PolyEnvelope1sR, r::T, x::T) where {T} = 
      (ForwardDiff.derivative(x -> evaluate(env, x), r), 
       zero(T),)

# ----------------------------

struct PolyEnvelope2sX{T}
   x1::T 
   x2::T 
   p1::Int 
   p2::Int
   s::T 
   # ------- 
   meta::Dict{String, Any}
end 

function PolyEnvelope2sX(x1, x2, p1, p2) 
   if x1 == x2 
      error("x1 and x2 must be different!")
   end
   if x1 > x2 
      @warn("swapping x1, x2 to ensure x1 < x2")
      x1, x2 = x2, x1
      p1, p2 = p2, p1
   end
   s = 1 / (abs(x2 - x1)/2)^(p1+p2)
   PolyEnvelope2sX(x1, x2, p1, p2, s, Dict{String, Any}())
end 


function evaluate(env::PolyEnvelope2sX, r::T, x::T) where T 
   x1, x2 = env.x1, env.x2
   p1, p2 = env.p1, env.p2
   s = env.s

   if !(x1 < x < x2)
      return zero(T)
   end

   return s * (x-x1)^p1 * (x2-x)^p2
end


evaluate_d(env::PolyEnvelope2sX, r::T, x::T) where T = 
    (zero(T), ForwardDiff.derivative(x -> evaluate(env, x), x))

    

