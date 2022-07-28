using ForwardDiff
using Random
using Distributions
include("DiagonalHessianApproximation.jl")

Random.seed!(123)
d = Normal(0.0,1.0)
dnoise = Normal(0.0,0.1)

# Points
x0 = rand(d,3)
x1 = x0 + rand(dnoise,3)

# Test functions
f(x) = x[1]^2 + x[2]^2 * x[3]^2
g(x) = exp(x[1]) + x[2] + cos(x[3])
h(x) = x[1]^2 * x[2] * x[3]^3

############################
# Test DiagonalQN
############################

# Exact initialization
Bf = DiagonalQN(diag(ForwardDiff.hessian(f,x0)))
Bg = DiagonalQN(diag(ForwardDiff.hessian(g,x0)))
Bh = DiagonalQN(diag(ForwardDiff.hessian(h,x0)))

# Update the approximations
push!(Bf,
      x1-x0,
      ForwardDiff.gradient(f,x1)-ForwardDiff.gradient(f,x0))
push!(Bg,
      x1-x0,
      ForwardDiff.gradient(g,x1)-ForwardDiff.gradient(g,x0))
push!(Bh,
      x1-x0,
      ForwardDiff.gradient(h,x1)-ForwardDiff.gradient(h,x0))


println("\nTEST DiagonalQN\n")
println("True f :")
println(diag(ForwardDiff.hessian(f,x1)))
println("Approx f :")
println(diag(Matrix(Bf)))
println("True g :")
println(diag(ForwardDiff.hessian(g,x1)))
println("Approx g :")
println(diag(Matrix(Bg)))
println("True h :")
println(diag(ForwardDiff.hessian(h,x1)))
println("Approx h :")
println(diag(Matrix(Bh)))

############################
# Test SpectralGradient
############################

# Exact initialization
Bf = SpectralGradient(diag(ForwardDiff.hessian(f,x0)))
Bg = SpectralGradient(diag(ForwardDiff.hessian(g,x0)))
Bh = SpectralGradient(diag(ForwardDiff.hessian(h,x0)))

# Update the approximations
push!(Bf,
      x1-x0,
      ForwardDiff.gradient(f,x1)-ForwardDiff.gradient(f,x0))
push!(Bg,
      x1-x0,
      ForwardDiff.gradient(g,x1)-ForwardDiff.gradient(g,x0))
push!(Bh,
      x1-x0,
      ForwardDiff.gradient(h,x1)-ForwardDiff.gradient(h,x0))

println("\nTEST SpectralGradient\n")
println("True f :")
println(diag(ForwardDiff.hessian(f,x1)))
println("Approx f :")
println(diag(Matrix(Bf)))
println("True g :")
println(diag(ForwardDiff.hessian(g,x1)))
println("Approx g :")
println(diag(Matrix(Bg)))
println("True h :")
println(diag(ForwardDiff.hessian(h,x1)))
println("Approx h :")
println(diag(Matrix(Bh)))

############################
# DiagonalModifiedSR1 with u = s
############################

# Exact initialization
Bf = DiagonalModifiedSR1(diag(ForwardDiff.hessian(f,x0)))
Bg = DiagonalModifiedSR1(diag(ForwardDiff.hessian(g,x0)))
Bh = DiagonalModifiedSR1(diag(ForwardDiff.hessian(h,x0)))

# Update the approximations
push!(Bf,
      x1-x0,
      ForwardDiff.gradient(f,x1)-ForwardDiff.gradient(f,x0),
      ForwardDiff.gradient(f,x1)+ForwardDiff.gradient(f,x0),
      f(x0)-f(x1),
      x1-x0)
push!(Bg,
      x1-x0,
      ForwardDiff.gradient(g,x1)-ForwardDiff.gradient(g,x0),
      ForwardDiff.gradient(g,x1)+ForwardDiff.gradient(g,x0),
      g(x0)-g(x1),
      x1-x0)
push!(Bh,
      x1-x0,
      ForwardDiff.gradient(h,x1)-ForwardDiff.gradient(h,x0),
      ForwardDiff.gradient(h,x1)+ForwardDiff.gradient(h,x0),
      h(x0)-h(x1),
      x1-x0)

println("\nTEST SpectralGradient with u = s\n")
println("True f :")
println(diag(ForwardDiff.hessian(f,x1)))
println("Approx f :")
println(diag(Matrix(Bf)))
println("True g :")
println(diag(ForwardDiff.hessian(g,x1)))
println("Approx g :")
println(diag(Matrix(Bg)))
println("True h :")
println(diag(ForwardDiff.hessian(h,x1)))
println("Approx h :")
println(diag(Matrix(Bh)))

############################
# DiagonalModifiedSR1 with u = y
############################

# Exact initialization
Bf = DiagonalModifiedSR1(diag(ForwardDiff.hessian(f,x0)))
Bg = DiagonalModifiedSR1(diag(ForwardDiff.hessian(g,x0)))
Bh = DiagonalModifiedSR1(diag(ForwardDiff.hessian(h,x0)))

# Update the approximations
push!(Bf,
      x1-x0,
      ForwardDiff.gradient(f,x1)-ForwardDiff.gradient(f,x0),
      ForwardDiff.gradient(f,x1)+ForwardDiff.gradient(f,x0),
      f(x0)-f(x1),
      ForwardDiff.gradient(f,x1)-ForwardDiff.gradient(f,x0))
push!(Bg,
      x1-x0,
      ForwardDiff.gradient(g,x1)-ForwardDiff.gradient(g,x0),
      ForwardDiff.gradient(g,x1)+ForwardDiff.gradient(g,x0),
      g(x0)-g(x1),
      ForwardDiff.gradient(g,x1)-ForwardDiff.gradient(g,x0))
push!(Bh,
      x1-x0,
      ForwardDiff.gradient(h,x1)-ForwardDiff.gradient(h,x0),
      ForwardDiff.gradient(h,x1)+ForwardDiff.gradient(h,x0),
      h(x0)-h(x1),
      ForwardDiff.gradient(h,x1)-ForwardDiff.gradient(h,x0))

println("\nTEST SpectralGradient with u = s\n")
println("True f :")
println(diag(ForwardDiff.hessian(f,x1)))
println("Approx f :")
println(diag(Matrix(Bf)))
println("True g :")
println(diag(ForwardDiff.hessian(g,x1)))
println("Approx g :")
println(diag(Matrix(Bg)))
println("True h :")
println(diag(ForwardDiff.hessian(h,x1)))
println("Approx h :")
println(diag(Matrix(Bh)))

############################
# DiagonalModifiedSR1 with u = ∇f(x_k)
############################

# Exact initialization
Bf = DiagonalModifiedSR1(diag(ForwardDiff.hessian(f,x0)))
Bg = DiagonalModifiedSR1(diag(ForwardDiff.hessian(g,x0)))
Bh = DiagonalModifiedSR1(diag(ForwardDiff.hessian(h,x0)))

# Update the approximations
push!(Bf,
      x1-x0,
      ForwardDiff.gradient(f,x1)-ForwardDiff.gradient(f,x0),
      ForwardDiff.gradient(f,x1)+ForwardDiff.gradient(f,x0),
      f(x0)-f(x1),
      ForwardDiff.gradient(f,x0))
push!(Bg,
      x1-x0,
      ForwardDiff.gradient(g,x1)-ForwardDiff.gradient(g,x0),
      ForwardDiff.gradient(g,x1)+ForwardDiff.gradient(g,x0),
      g(x0)-g(x1),
      ForwardDiff.gradient(g,x0))
push!(Bh,
      x1-x0,
      ForwardDiff.gradient(h,x1)-ForwardDiff.gradient(h,x0),
      ForwardDiff.gradient(h,x1)+ForwardDiff.gradient(h,x0),
      h(x0)-h(x1),
      ForwardDiff.gradient(h,x0))

println("\nTEST SpectralGradient with u = s\n")
println("True f :")
println(diag(ForwardDiff.hessian(f,x1)))
println("Approx f :")
println(diag(Matrix(Bf)))
println("True g :")
println(diag(ForwardDiff.hessian(g,x1)))
println("Approx g :")
println(diag(Matrix(Bg)))
println("True h :")
println(diag(ForwardDiff.hessian(h,x1)))
println("Approx h :")
println(diag(Matrix(Bh)))