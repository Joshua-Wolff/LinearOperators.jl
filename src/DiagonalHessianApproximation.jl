using LinearAlgebra
using LinearOperators

function mulSquareOpDiagonal!(res, d, v, α, β::T) where {T <: Real}
  if β == zero(T)
    res .= α .* d .* v
  else
    res .= α .* d .* v .+ β .* res
  end
end

abstract type AbstractDiagonalQuasiNewtonOperator{T} <: AbstractLinearOperator{T} end

"""
Implementation of the diagonal quasi-Newton approximation described in

Andrei, N. 
A diagonal quasi-Newton updating method for unconstrained optimization. 
https://doi.org/10.1007/s11075-018-0562-7
"""

mutable struct DiagonalQN{T <: Real, I <: Integer, V <: AbstractVector{T}} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::V # Diagonal of the operator
  Bs::V # Preallocate a vector used in push! function
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

DiagonalQN(d::AbstractVector{T}) where {T <: Real} = 
  DiagonalQN(
    d,
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
  B::DiagonalQN{T,I,V},
  s::V,
  y::V
  ) where {T <: Real, I <: Integer, V <: AbstractVector{T}}
  trA2 = 0
  for i in eachindex(s)
    trA2 += s[i]^4
  end
  sT_s = dot(s,s)
  sT_y = dot(s,y)
  B.Bs .= B*s
  sT_B_s = dot(s,B.Bs)
  if trA2 == 0
    error("Cannot divide by zero and trA2 = 0")
  end
  q = (sT_y + sT_s - sT_B_s)/trA2
  B.d .+= q .* s.^2 .- 1
end

"""
Implementation of a spectral gradient quasi-Newton approximation described in

Birgin, E. G., Martínez, J. M., & Raydan, M. 
Spectral Projected Gradient Methods: Review and Perspectives. 
https://doi.org/10.18637/jss.v060.i03
"""

mutable struct SpectralGradient{T <: Real, I <: Integer, V <: AbstractVector{T}} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::T # Diagonal coefficient of the operator (multiple of the identity)
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

SpectralGradient(d::T, n::I, V) where {T <: Real, I <: Integer} = 
  SpectralGradient(
    d,
    n,
    n,
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
    V(undef,0),
    V(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::SpectralGradient{T,I,V},
  s::V,
  y::V
  ) where {T <: Real, I <: Integer, V <: AbstractVector{T}}
  if all(s .== 0)
    error("Cannot divide by zero and s .= 0")
  end
  B.d = dot(s,y)/dot(s,s)
end