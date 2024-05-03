
abstract type AbstractEnvelope end

struct PolyEnvelope1sR{T}
   rcut::T
   p::Int 
   # ------- 
   meta::Dict{String, Any}
end 


PolyEnvelope1sR(rcut, p) = 
   PolyEnvelope1sR(rcut, p, Dict{String, Any}())

function evaluate(env::PolyEnvelope1sR, r::T) where T 
   if r >= env.rcut 
      return zero(T)
   end
   p = env.p 
   # return r^(-p) - env.rcut^(-p) - p*(env.rcut^(-p-1))*(r - env.rcut)
   return ( (r/env.rcut)^(-p) - 1.0) * (1 - r / env.rcut)
end

evaluate_d(env::PolyEnvelope1sR, r) = 
   ForwardDiff.derivative(x -> evaluate(env, x), r)

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


function evaluate(env::PolyEnvelope2sX, x::T) where T 
   x1, x2 = env.x1, env.x2
   p1, p2 = env.p1, env.p2
   s = env.s

   if !(x1 < x < x2)
      return zero(T)
   end

   return s * (x-x1)^p1 * (x2-x)^p2
end


evaluate_d(env::PolyEnvelope2sX, x::T) where T = 
   ForwardDiff.derivative(x -> evaluate(env, x), x)


