module Metrics
	
using Devectorize

export 
	# generic types/functions
	GeneralizedMetric, 
	Metric,
	
	# distance classes
	Euclidean,
	SqEuclidean,
	Chebyshev,
	Cityblock,
	Minkowski,
	Hamming,
	CosineDist,
	CorrDist,
	ChiSqDist,
	KLDivergence,
	JSDivergence,
	
	# convenient functions
	euclidean,
	sqeuclidean,
	chebyshev,
	cityblock,
	minkowski,
	hamming,
	cosine_dist,
	corr_dist,
	chisq_dist,
	kl_divergence,
	js_divergence,

	# generic functions 
	result_type,
	colwise,
	pairwise,
	evaluate


###########################################################
#
#	Metric types
#
###########################################################

# a premetric is a function d that satisfies:
#
#	d(x, y) >= 0
#	d(x, x) = 0
#
abstract PreMetric

# a semimetric is a function d that satisfies:
#
#	d(x, y) >= 0
#	d(x, x) = 0
#	d(x, y) = d(y, x)
#
abstract SemiMetric <: PreMetric

# a metric is a semimetric that satisfies triangle inequality:
#
#	d(x, y) + d(y, z) >= d(x, z)
#
abstract Metric <: SemiMetric

type Euclidean <: Metric end
type SqEuclidean <: SemiMetric end
type Chebyshev <: Metric end
type Cityblock <: Metric end

type Minkowski <: Metric 
	p::Real
end

type Hamming <: Metric end
type CosineDist <: SemiMetric end
type CorrDist <: SemiMetric end
	
type ChiSqDist <: SemiMetric end
type KLDivergence <: PreMetric end
type JSDivergence <: SemiMetric end
	


###########################################################
#
#	result types
#
###########################################################

result_type(::PreMetric, T1::Type, T2::Type) = promote_type(T1, T2)
result_type(::Hamming, T1::Type, T2::Type) = Int


###########################################################
#
#	Generic colwise and pairwise evaluation
#
###########################################################

function colwise!(r::Array, metric::PreMetric, a::Vector, b::Matrix)
	for j = 1 : n
		r[j] = evaluate(metric, a, b[:,j])
	end
end

function colwise!(r::Array, metric::PreMetric, a::Matrix, b::Vector)
	for j = 1 : n
		r[j] = evaluate(metric, a[:,j], b)
	end
end

function colwise!(r::Array, metric::PreMetric, a::Matrix, b::Matrix)
	for j = 1 : n
		r[j] = evaluate(metric, a[:,j], b[:,j])
	end
end

function colwise!(r::Array, metric::SemiMetric, a::Matrix, b::Vector)
	colwise!(r, metric, b, a)
end

function colwise(metric::PreMetric, a::Matrix, b::Matrix)
	n = size(a, 2)
	if size(b, 2) != n
		throw(ArgumentError("The number of columns of a and b must match."))
	end

	r = Array(result_type(metric, eltype(a), eltype(b)), n)
	colwise!(r, metric, a, b)
	return r
end

function colwise(metric::PreMetric, a::Vector, b::Matrix)
	n = size(b, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), n)
	colwise!(r, metric, a, b)
	return r
end

function colwise(metric::PreMetric, a::Matrix, b::Vector)
	n = size(a, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), n)
	colwise!(r, metric, a, b)
	return r
end


function pairwise!(r::Matrix, metric::PreMetric, a::Matrix, b::Matrix)
	for j = 1 : size(b, 2)
		bj = b[:,j]
		for i = 1 : size(a, 2)
			r[i,j] = evaluate(metric, a[:,i], bj)
		end
	end
end

function pairwise!(r::Matrix, metric::PreMetric, a::Matrix)
	pairwise!(r, metric, a, a)
end


# faster evaluation by leveraging the properties of semi-metrics
function pairwise!(r::Matrix, metric::SemiMetric, a::Matrix)
	n = size(a, 2)
	for j = 1 : n
		for i = 1 : j-1
			a[i,j] = a[j,i]
		end
		a[j,j] = 0
		bj = b[:,j]
		for i = j+1 : n
			a[i,j] = evaluate(metric, a[:,i], bj)
		end
	end
end

function pairwise(metric::PreMetric, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), (m, n))
	pairwise!(r, metric, a, b)
	return r
end

function pairwise(metric::PreMetric, a::Matrix)
	n = size(a, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), (n, n))
	pairwise!(r, metric, a)
	return r
end

function pairwise(metric::SemiMetric, a::Matrix)
	n = size(a, 2)
	r = Array(result_type(metric, eltype(a), eltype(a)), (n, n))
	pairwise!(r, metric, a)
	return r
end


###########################################################
#
#	Specialized distances
#
###########################################################

# SqEuclidean

function evaluate(dist::SqEuclidean, a::Vector, b::Vector)
	@devec r = sum(sqr(a - b))
	return r
end

sqeuclidean(a::Vector, b::Vector) = evaluate(SqEuclidean(), a, b)

function colwise!(r::Array, dist::SqEuclidean, a::Matrix, b::Matrix)
	@devec r[:] = sum(sqr(a - b), 1)
end

function colwise!(r::Array, dist::SqEuclidean, a::Vector, b::Matrix)
	for j = 1 : size(b, 2)
		@devec r[j] = sum(sqr(a - b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::SqEuclidean, a::Matrix, b::Matrix)
	At_mul_B(r, a, b)
	@devec sa2 = sum(sqr(a), 1)
	@devec sb2 = sum(sqr(b), 1)

	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
		end
	end
end

function pairwise!(r::Matrix, dist::SqEuclidean, a::Matrix)
	At_mul_B(r, a, a)
	@devec sa2 = sum(sqr(a), 1)

	n = size(a, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			r[i,j] = sa2[i] + sa2[j] - 2 * r[i,j]
		end
	end
end

# Euclidean

function evaluate(dist::Euclidean, a::Vector, b::Vector)
	@devec r = sum(sqr(a - b))
	return sqrt(r)
end

euclidean(a::Vector, b::Vector) = evaluate(Euclidean(), a, b)

function colwise!(r::Array, dist::Euclidean, a::Matrix, b::Matrix)
	@devec r[:] = sqrt(sum(sqr(a - b), 1))
end

function colwise!(r::Array, dist::Euclidean, a::Vector, b::Matrix)
	for j = 1 : size(b, 2)
		@devec r[j] = sum(sqr(a - b[:,j]))
		r[j] = sqrt(r[j])
	end
end

function pairwise!(r::Matrix, dist::Euclidean, a::Matrix, b::Matrix)
	pairwise!(r, SqEuclidean(), a, b)
	@devec r[:] = sqrt(max(r, 0))
end

function pairwise!(r::Matrix, dist::Euclidean, a::Matrix)
	pairwise!(r, SqEuclidean(), a)
	@devec r[:] = sqrt(max(r, 0))
end


# Cityblock

function evaluate(dist::Cityblock, a::Vector, b::Vector)
	@devec r = sum(abs(a - b))
	return r
end

cityblock(a::Vector, b::Vector) = evaluate(Cityblock(), a, b)

function colwise!(r::Array, dist::Cityblock, a::Matrix, b::Matrix)
	@devec r[:] = sum(abs(a - b), 1)
end

function colwise!(r::Array, dist::Cityblock, a::Vector, b::Matrix)
	for j = 1 : size(b, 2)
		@devec r[j] = sum(abs(a - b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::Cityblock, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			@devec r[i,j] = sum(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::Matrix, dist::Cityblock, a::Matrix)
	n = size(a, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec r[i,j] = sum(abs(a[:,i] - a[:,j]))
		end
	end
end


# Chebyshev

function evaluate(dist::Chebyshev, a::Vector, b::Vector)
	@devec r = max(abs(a - b))
	return r
end

chebyshev(a::Vector, b::Vector) = evaluate(Chebyshev(), a, b)

function colwise!(r::Array, dist::Chebyshev, a::Matrix, b::Matrix)
	@devec r[:] = max(abs(a - b), (), 1)
end

function colwise!(r::Array, dist::Chebyshev, a::Vector, b::Matrix)
	for j = 1 : size(b, 2)
		@devec r[j] = max(abs(a - b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::Chebyshev, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			@devec r[i,j] = max(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::Matrix, dist::Chebyshev, a::Matrix)
	n = size(a, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec r[i,j] = max(abs(a[:,i] - a[:,j]))
		end
	end
end


# Minkowski

function evaluate(dist::Minkowski, a::Vector, b::Vector)
	p = dist.p
	@devec r = sum(abs(a - b) .^ p)
	return r ^ (1 / p)
end

minkowski(a::Vector, b::Vector, p::Real) = evaluate(Minkowski(p), a, b)

function colwise!(r::Array, dist::Minkowski, a::Matrix, b::Matrix)
	p = dist.p
	inv_p = 1 / p
	@devec r[:] = sum(abs(a - b) .^ p, 1) .^ inv_p
end

function colwise!(r::Array, dist::Minkowski, a::Vector, b::Matrix)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : size(b, 2)
		@devec r[j] = sum(abs(a - b[:,j]) .^ p)
		r[j] = r[j] ^ inv_p
	end
end

function pairwise!(r::Matrix, dist::Minkowski, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		for i = 1 : m
			@devec t = sum(abs(a[:,i] - b[:,j]) .^ p)
			r[i,j] = t ^ inv_p
		end
	end
end

function pairwise!(r::Matrix, dist::Minkowski, a::Matrix)
	n = size(a, 2)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec t = sum(abs(a[:,i] - a[:,j]) .^ p)
			r[i,j] = t ^ inv_p
		end
	end
end


# Hamming

function evaluate(dist::Hamming, a::Vector, b::Vector)
	sum(a .!= b)
end

hamming(a::Vector, b::Vector) = evaluate(Hamming(), a, b)

function colwise!(r::Array, dist::Hamming, a::Matrix, b::Matrix)
	m, n = size(a)
	if size(b) != (m, n)
		throw(ArgumentError("The sizes of a and b must match."))
	end
	
	for j = 1 : n
		d::Int = 0
		for i = 1 : m
			if (a[i,j] != b[i,j]) 
				d += 1
			end
		end
		r[j] = d
	end
end

function colwise!(r::Array, dist::Hamming, a::Vector, b::Matrix)
	m, n = size(b)
	for j = 1 : n
		d::Int = 0
		for i = 1 : m
			if (a[i] != b[i,j]) 
				d += 1
			end
		end
		r[j] = d
	end
end

function pairwise!(r::Matrix, dist::Hamming, a::Matrix, b::Matrix)
	m, na = size(a)
	nb = size(b, 2)
	for j = 1 : nb
		for i = 1 : na
			d::Int = 0
			for k = 1 : m
				if a[k,i] != b[k,j]
					d += 1
				end	
			end
			r[i,j] = d
		end
	end
end

function pairwise!(r::Matrix, dist::Hamming, a::Matrix)
	m, n = size(a)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			d::Int = 0
			for k = 1 : m
				if a[k,i] != a[k,j]
					d += 1
				end
			end
			r[i,j] = d
		end
	end
end


# Cosine dist

function evaluate(dist::CosineDist, a::Vector, b::Vector)
	max(1 - dot(a, b) / (norm(a) * norm(b)), 0)
end

cosine_dist(a::Vector, b::Vector) = evaluate(CosineDist(), a, b)

function colwise!(r::Array, dist::CosineDist, a::Matrix, b::Matrix)
	@devec begin
		ra = sum(sqr(a), 1)
		rb = sum(sqr(b), 1)
		ra[:] = sqrt(ra)
		rb[:] = sqrt(rb)
		ab = sum(a .* b, 1)
		r[:] = max(1 - ab ./ (ra .* rb), 0)
	end
end

function colwise!(r::Array, dist::CosineDist, a::Vector, b::Matrix)
	@devec begin
		ra = sqrt(sum(sqr(a)))
		rb = sum(sqr(b), 1)
		rb[:] = sqrt(rb)
	end
	ab = At_mul_B(b, a)
	@devec r[:] = max(1 - ab ./ (ra .* rb), 0)
end

function pairwise!(r::Matrix, dist::CosineDist, a::Matrix, b::Matrix)
	At_mul_B(r, a, b)
	@devec begin
		ra = sum(sqr(a), 1)
		rb = sum(sqr(b), 1)
		ra[:] = sqrt(ra)
		rb[:] = sqrt(rb)
	end
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			r[i,j] = max(1 - r[i,j] / (ra[i] * rb[j]), 0)
		end
	end
end

function pairwise!(r::Matrix, dist::CosineDist, a::Matrix)
	At_mul_B(r, a, a)
	@devec begin
		ra = sum(sqr(a), 1)
		ra[:] = sqrt(ra)
	end
	n = size(a, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			r[i,j] = max(1 - r[i,j] / (ra[i] * ra[j]), 0)
		end
	end
end

# Correlation Dist

function evaluate(dist::CorrDist, a::Vector, b::Vector)
	cosine_dist(a - mean(a), b - mean(b))
end

corr_dist(a::Vector, b::Vector) = evaluate(CorrDist(), a, b)

function colwise!(r::Array, dist::CorrDist, a::Matrix, b::Matrix)
	a_ = bsxfun(-, a, mean(a, 1))
	b_ = bsxfun(-, b, mean(b, 1))
	colwise!(r, CosineDist(), a_, b_)
end

function colwise!(r::Array, dist::CorrDist, a::Vector, b::Matrix)
	a_ = a - mean(a)
	b_ = bsxfun(-, b, mean(b, 1))
	colwise!(r, CosineDist(), a_, b_)
end

function pairwise!(r::Matrix, dist::CorrDist, a::Matrix, b::Matrix)
	a_ = bsxfun(-, a, mean(a, 1))
	b_ = bsxfun(-, b, mean(b, 1))
	pairwise!(r, CosineDist(), a_, b_)
end

function pairwise!(r::Matrix, dist::CorrDist, a::Matrix)
	a_ = bsxfun(-, a, mean(a, 1))
	pairwise!(r, CosineDist(), a_)
end


# Chi-square distance

function evaluate(dist::ChiSqDist, a::Vector, b::Vector)
	@devec r = sum(sqr(a - b) ./ (a + b))
	return r
end

chisq_dist(a::Vector, b::Vector) = evaluate(ChiSqDist(), a, b)

function colwise!(r::Array, dist::ChiSqDist, a::Matrix, b::Matrix)
	@devec r[:] = sum(sqr(a - b) ./ (a + b), 1)
end

function colwise!(r::Array, dist::ChiSqDist, a::Vector, b::Matrix)
	for j = 1 : size(b, 2)
		@devec r[j] = sum(sqr(a - b[:,j]) ./ (a + b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::ChiSqDist, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			@devec r[i,j] = sum(sqr(a[:,i] - b[:,j]) ./ (a[:,i] + b[:,j]))
		end
	end
end

function pairwise!(r::Matrix, dist::ChiSqDist, a::Matrix)
	n = size(a, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec r[i,j] = sum(sqr(a[:,i] - a[:,j]) ./ (a[:,i] + a[:,j]))
		end
	end
end


end # module end
	
	