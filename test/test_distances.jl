# Unit tests for Distances

using Test
using Distances

# facilities to simplify testing

all_approx(tol, a, b) = all(abs(a - b) .< tol)
all_approx(a, b) = all_approx(1.0e-12, a, b)

macro test_approx(a, b)
	:( @test size($a) == size($b) && all_approx($a, $b) )
end

# tests

d = 6
n = 10

a = rand(d, n)
b = rand(d, n)

a_int = mod(rand(Int, (d, n)), 3)  # for testing hamming
b_int = mod(rand(Int, (d, n)), 3)

a1 = a[:,1]
b1 = b[:,1]

a1_int = a_int[:,1]
b1_int = b_int[:,1]


println("testing euclidean distance...")

eucdist_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		eucdist_r[i, j] = norm(a[:,i] - b[:,j])
	end
end

@test_approx euclidean_dist(a1, b1) eucdist_r[1,1]
@test_approx euclidean_dist(a1, b)  vec(eucdist_r[1,:])
@test_approx euclidean_dist(a,  b1) vec(eucdist_r[:,1])
@test_approx euclidean_dist(a,  b)  diag(eucdist_r)


println("testing sq-euclidean distance...")

sqeucdist_r = eucdist_r .^ 2

@test_approx sqeuclidean_dist(a1, b1) sqeucdist_r[1,1]
@test_approx sqeuclidean_dist(a1, b)  vec(sqeucdist_r[1,:])
@test_approx sqeuclidean_dist(a,  b1) vec(sqeucdist_r[:,1])
@test_approx sqeuclidean_dist(a,  b)  diag(sqeucdist_r)


println("testing cityblock distance ...")

cityblkdist_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		cityblkdist_r[i, j] = norm(a[:,i] - b[:,j], 1)
	end
end

@test_approx cityblock_dist(a1, b1) cityblkdist_r[1,1]
@test_approx cityblock_dist(a1, b)  vec(cityblkdist_r[1,:])
@test_approx cityblock_dist(a,  b1) vec(cityblkdist_r[:,1])
@test_approx cityblock_dist(a,  b)  diag(cityblkdist_r)


println("testing chebyshev distance...")

chebydist_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		chebydist_r[i, j] = norm(a[:,i] - b[:,j], Inf)
	end
end

@test_approx chebyshev_dist(a1, b1) chebydist_r[1,1]
@test_approx chebyshev_dist(a1, b)  vec(chebydist_r[1,:])
@test_approx chebyshev_dist(a,  b1) vec(chebydist_r[:,1])
@test_approx chebyshev_dist(a,  b)  diag(chebydist_r)


println("testing minkowski distance...")

p = 2.6
minkowdist_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		minkowdist_r[i, j] = norm(a[:,i] - b[:,j], p)
	end
end

@test_approx minkowski_dist(a1, b1, p) minkowdist_r[1,1]
@test_approx minkowski_dist(a1, b,  p) vec(minkowdist_r[1,:])
@test_approx minkowski_dist(a,  b1, p) vec(minkowdist_r[:,1])
@test_approx minkowski_dist(a,  b,  p) diag(minkowdist_r)


println("testing hamming distance...")

hamdist_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		hamdist_r[i, j] = sum(a_int[:,i] .!= b_int[:,j])
	end
end

@test_approx hamming_dist(a1_int, b1_int) hamdist_r[1,1]
@test_approx hamming_dist(a1_int, b_int)  vec(hamdist_r[1,:])
@test_approx hamming_dist(a_int,  b1_int) vec(hamdist_r[:,1])
@test_approx hamming_dist(a_int,  b_int)  diag(hamdist_r)


println("testing cosine distance...")

cosdist_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		ai = a[:, i]
		bj = b[:, j]
		cosdist_r[i, j] = sum(ai .* bj) / (norm(ai) * norm(bj))
	end
end

@test_approx cosine_dist(a1, b1) cosdist_r[1,1]
@test_approx cosine_dist(a1, b)  vec(cosdist_r[1,:])
@test_approx cosine_dist(a,  b1) vec(cosdist_r[:,1])
@test_approx cosine_dist(a,  b)  diag(cosdist_r)


println("testing K-L divergence...")

kldiv_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		ai = a[:, i]
		bj = b[:, j]
		kldiv_r[i, j] = sum(ai .* log(ai ./ bj))
	end
end

@test_approx kl_div(a1, b1) kldiv_r[1,1]
@test_approx kl_div(a1, b)  vec(kldiv_r[1,:])
@test_approx kl_div(a,  b1) vec(kldiv_r[:,1])
@test_approx kl_div(a,  b)  diag(kldiv_r)


println("testing J-S divergence...")

jsdiv_r = zeros(n, n)
for j = 1 : n
	for i = 1 : n
		ai = a[:,i]
		bj = b[:,j]
		v = (ai + bj) * 0.5
		jsdiv_r[i,j] = (kl_div(ai, v) + kl_div(bj, v)) / 2
	end
end

@test_approx js_div(a1, b1) jsdiv_r[1,1]
@test_approx js_div(a1, b)  vec(jsdiv_r[1,:])
@test_approx js_div(a,  b1) vec(jsdiv_r[:,1])
@test_approx js_div(a,  b)  diag(jsdiv_r)




