# Unit tests for Distances

using Test
using Metrics

# helpers

is_approx(a::Number, b::Number, tol::Number) = abs(a - b) < tol
all_approx(a::Array, b::Array, tol::Number) = size(a) == size(b) && all(abs(a - b) .< tol)


# test individual metrics

x = [4., 5., 6., 7.]
y = [3., 9., 8., 1.]

a = [1., 2., 1., 3., 2., 1.]
b = [1., 3., 0., 2., 2., 0.]

@test sqeuclidean(x, x) == 0.
@test sqeuclidean(x, y) == 57.

@test euclidean(x, x) == 0.
@test euclidean(x, y) == sqrt(57.)

@test cityblock(x, x) == 0.
@test cityblock(x, y) == 13.

@test chebyshev(x, x) == 0.
@test chebyshev(x, y) == 6.

@test minkowski(x, x, 2) == 0.
@test minkowski(x, y, 2) == sqrt(57.)

@test hamming(a, a) == 0
@test hamming(a, b) == 4

@test is_approx(cosine_dist(x, x), 0., 1.0e-14)
@test is_approx(cosine_dist(x, y), 1. - 112. / sqrt(19530.), 1.0e-14)

@test is_approx(corr_dist(x, x), 0., 1.0e-14)
@test corr_dist(x, y) == cosine_dist(x - mean(x), y - mean(y))

# test column-wise metrics

m = 5
n = 8
X = rand(m, n)
Y = rand(m, n)
A = rand(1:3, m, n)
B = rand(1:3, m, n)

macro test_colwise(dist, x, y, tol)
	quote
		local n = size($x, 2)
		r1 = zeros(n)
		r2 = zeros(n)
		r3 = zeros(n)
		for j = 1 : n
			r1[j] = evaluate($dist, ($x)[:,j], ($y)[:,j])
			r2[j] = evaluate($dist, ($x)[:,1], ($y)[:,j])
			r3[j] = evaluate($dist, ($x)[:,j], ($y)[:,1])
		end 
		@test all_approx(colwise($dist, $x, $y),        r1, $tol)
		@test all_approx(colwise($dist, ($x)[:,1], $y), r2, $tol)
		@test all_approx(colwise($dist, $x, ($y)[:,1]), r3, $tol)
	end
end

@test_colwise SqEuclidean() X Y 1.0e-14
@test_colwise Euclidean() X Y 1.0e-14
@test_colwise Cityblock() X Y 1.0e-14
@test_colwise Chebyshev() X Y 1.0e-16
@test_colwise Minkowski(2.5) X Y 1.0e-13
@test_colwise Hamming() A B 1.0e-16

@test_colwise CosineDist() X Y 1.0e-14
@test_colwise CorrDist() X Y 1.0e-14


# test pairwise metrics

nx = 6
ny = 8

X = rand(m, nx)
Y = rand(m, ny)
A = rand(1:3, m, nx)
B = rand(1:3, m, ny)

macro test_pairwise(dist, x, y, tol)
	quote
		local nx = size($x, 2)
		local ny = size($y, 2)
		rxy = zeros(nx, ny)
		rxx = zeros(nx, nx)
		for j = 1 : ny, i = 1 : nx
			rxy[i, j] = evaluate($dist, ($x)[:,i], ($y)[:,j])
		end
		for j = 1 : nx, i = 1 : nx
			rxx[i, j] = evaluate($dist, ($x)[:,i], ($x)[:,j])
		end
		#println("rxy = ", rxy)
		#println("res = ", pairwise($dist, $x, $y))
		@test all_approx(pairwise($dist, $x, $y), rxy, $tol)
		@test all_approx(pairwise($dist, $x), rxx, $tol)
	end
end

@test_pairwise SqEuclidean() X Y 1.0e-14
@test_pairwise Euclidean() X Y 1.0e-14
@test_pairwise Cityblock() X Y 1.0e-14
@test_pairwise Chebyshev() X Y 1.0e-16
@test_pairwise Minkowski(2.5) X Y 1.0e-14
@test_pairwise Hamming() A B 1.0e-16

@test_pairwise CosineDist() X Y 1.0e-14 
@test_pairwise CorrDist() X Y 1.0e-14

