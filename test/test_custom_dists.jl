# Unit testing of generic functions on user-defined distances

using Distance
using Test
import Distance.evaluate

type MyEuclidean <: Metric
end

function evaluate(dist::MyEuclidean, x::AbstractVector, y::AbstractVector)
	sqrt(sum((x - y).^2))	
end

m = 5
nx = 8
ny = 10

# single-vector

x = rand(m)
y = rand(m)

dist = MyEuclidean()

r = evaluate(dist, x, y)
r0 = euclidean(x, y)

@test abs(r - r0) < 1.0e-14


# column wise

x = rand(m, nx)
y = rand(m, nx)

r = colwise(dist, x, y)
r0 = colwise(Euclidean(), x, y)

@test size(r) == (nx,)
@test all(abs(r - r0) .< 1.0e-14)

# pairwise

x = rand(m, nx)
y = rand(m, ny)

r = pairwise(dist, x)
r0 = pairwise(Euclidean(), x)

@test size(r) == (nx, nx)
@test all(abs(r - r0) .< 1.0e-13)

r = pairwise(dist, x, y)
r0 = pairwise(Euclidean(), x, y)

@test size(r) == (nx, ny)
@test all(abs(r - r0) .< 1.0e-13)

