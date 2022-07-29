using LinearAlgebra
using LinearOperators

############ function to define products

function mulSquareOpDiagonal!(res, d, v, α, β::T) where {T <: Real}
  if β == zero(T)
    res .= α .* d .* v
  else
    res .= α .* d .* v .+ β .* res
  end
end

############ Abstract type

abstract type AbstractDiagonalQuasiNewtonOperator{T} <: AbstractLinearOperator{T} end

"""
Implementation a diagonal QN method coming from the following article : 
Andrei, N. 
A diagonal quasi-Newton updating method for unconstrained optimization. 
Numer Algor 81, 575–590 (2019). 
https://doi.org/10.1007/s11075-018-0562-7
"""
# core structure
mutable struct DiagonalQN{T <: Real, I <: Integer, V <: AbstractVector{T}} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::V # Diagonal of the operator matrix
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!
  tprod!
  ctprod!
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  Mv5::V
  Mtu5::V
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

# constructor
DiagonalQN(d::AbstractVector{T}) where {T <: Real} = 
  DiagonalQN(
    d,
    length(d),
    length(d),
    true, 
    true,  
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    0,
    0,
    0,
    true,
    true,
    typeof(d)(undef,0),
    typeof(d)(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::DiagonalQN{T,I},
  s::V,
  y::V
  ) where {T <: Real, I <: Integer, V <: AbstractVector{T}}
  trA2 = 0
  for i in eachindex(s)
    trA2 += s[i]^4
  end
  sT_s = dot(s,s)
  sT_y = dot(s,y)
  sT_B_s = dot(s,B*s)
  q = (sT_y + sT_s - sT_B_s)/trA2
  s2 = s.^2
  for i in eachindex(s)
    B.d[i] = B.d[i] + q * s2[i] - 1
  end
end

"""
Implementation a spectral gradient method coming from the following article : 
Birgin, E. G., Martínez, J. M., & Raydan, M. (2014). 
Spectral Projected Gradient Methods: Review and Perspectives. 
Journal of Statistical Software, 60(3), 1–21. 
https://doi.org/10.18637/jss.v060.i03
"""

# core structure
mutable struct SpectralGradient{T <: Real, I <: Integer, V <: AbstractVector{T}} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::V # Diagonal of the operator matrix
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!
  tprod!
  ctprod!
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  Mv5::V
  Mtu5::V
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

# constructor
SpectralGradient(d::AbstractVector{T}) where {T <: Real} = 
  SpectralGradient(
    d,
    length(d),
    length(d),
    true, 
    true,  
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    0,
    0,
    0,
    true,
    true,
    typeof(d)(undef,0),
    typeof(d)(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::SpectralGradient{T,I},
  s::V,
  y::V
  ) where {T <: Real, I <: Integer, V <: AbstractVector{T}}
  B.d .= dot(s,y)/dot(s,s) .* ones(length(s)) 
end

"""
Implementation a modified SR1 method coming from the following article : 
Farzin Modarres, Abu Hassan Malik, Wah June Leong,
Improved Hessian approximation with modified secant equations for symmetric rank-one method,
Journal of Computational and Applied Mathematics,
Volume 235, Issue 8,
2011,
Pages 2423-2431,
ISSN 0377-0427,
https://doi.org/10.1016/j.cam.2010.10.042.
"""

# core structure
mutable struct DiagonalModifiedSR1{T <: Real, I <: Integer, V <: AbstractVector{T}} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::V # Diagonal of the operator matrix
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!
  tprod!
  ctprod!
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  Mv5::V
  Mtu5::V
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

# constructor
DiagonalModifiedSR1(d::AbstractVector{T}) where {T <: Real} = 
  DiagonalModifiedSR1(
    d,
    length(d),
    length(d),
    true, 
    true,  
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β), 
    0,
    0,
    0,
    true,
    true,
    typeof(d)(undef,0),
    typeof(d)(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
# t = ∇f(x_k) + ∇f(x_{k+1})
# z = f(x_k) - f(x_{k+1})
# u ∈ {s,y,∇f(x_k)}
function push!(
  B::DiagonalModifiedSR1{T,I},
  s::V,
  y::V,
  t::V,
  z::T,
  u::V
  ) where {T <: Real, I <: Integer, V <: AbstractVector{T}}
  ψ = 2 * z + dot(t,s)
  yt = y + abs(ψ)/dot(s,u) * u
  delta = yt - B*s
  for i in 1:length(s)
    B.d[i] = B.d[i] + dot(delta,delta)/dot(delta,s)
  end 
end