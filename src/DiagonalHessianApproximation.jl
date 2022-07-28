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

############ Andrei 2018

# core structure
mutable struct DiagonalQN{T <: Real, I <: Integer} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::Vector{T} # Diagonal of the operator matrix
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
  Mv5::Vector{T}
  Mtu5::Vector{T}
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

# constructor
DiagonalQN(d::Vector{T}) where {T <: Real} = 
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
    Vector{T}(undef,0),
    Vector{T}(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::DiagonalQN{T,I},
  s::Vector{T},
  y::Vector{T},
  ) where {T <: Real, I <: Integer}
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

############ Spectral gradient

# core structure
mutable struct SpectralGradient{T <: Real, I <: Integer} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::Vector{T} # Diagonal of the operator matrix
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
  Mv5::Vector{T}
  Mtu5::Vector{T}
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

# constructor
SpectralGradient(d::Vector{T}) where {T <: Real} = 
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
    Vector{T}(undef,0),
    Vector{T}(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::SpectralGradient{T,I},
  s::Vector{T},
  y::Vector{T}
  ) where {T <: Real, I <: Integer}
  B.d .= dot(s,y)/dot(s,s) .* ones(length(s)) 
end

############ Modified Diagonal SR1

# core structure
mutable struct DiagonalModifiedSR1{T <: Real, I <: Integer} <: AbstractDiagonalQuasiNewtonOperator{T} 
  d::Vector{T} # Diagonal of the operator matrix
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
  Mv5::Vector{T}
  Mtu5::Vector{T}
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

# constructor
DiagonalModifiedSR1(d::Vector{T}) where {T <: Real} = 
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
    Vector{T}(undef,0),
    Vector{T}(undef,0),
    true)

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
# t = ∇f(x_k) + ∇f(x_{k+1})
# z = f(x_k) - f(x_{k+1})
# u ∈ {s,y,∇f(x_k)}
function push!(
  B::DiagonalModifiedSR1{T,I},
  s::Vector{T},
  y::Vector{T},
  t::Vector{T},
  z::T,
  u::Vector{T}
  ) where {T <: Real, I <: Integer}
  ψ = 2 * z + dot(t,s)
  yt = y + abs(ψ)/dot(s,u) * u
  delta = yt - B*s
  for i in 1:length(s)
    B.d[i] = B.d[i] + dot(delta,delta)/dot(delta,s)
  end 
end