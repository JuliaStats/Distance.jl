module Metrics
	
using DeExpr

export 
	# generic types/functions
	GeneralizedMetric, 
	Metric,
	cdist,
	pdist,
	
	# distance classes
	Euclidean,
	SqEuclidean,
	Chebyshev,
	Cityblock,
	Minkowski,
	Hamming,
	CosineDist,
	CorrDist,
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


function pairwise!(r::Array, metric::PreMetric, a::Matrix, b::Matrix)
	for j = 1 : size(b, 2)
		bj = b[:,j]
		for i = 1 : size(a, 2)
			r[i,j] = evaluate(metric, a[:,i], bj)
		end
	end
end

function pairwise!(r::Array, metric::PreMetric, a::Matrix)
	pairwise!(r, metric, a, a)
end


# faster evaluation by leveraging the properties of semi-metrics
function pairwise!(r::Array, metric::SemiMetric, a::Matrix)
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
	r = Array(result_type(metric, eltype(a), eltype(b)), (n, n))
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

function pairwise!(r::Array, dist::SqEuclidean, a::Matrix, b::Matrix)
	A_mul_Bt(r, a, b)
	@devec sa2 = sum(sqr(a), 1)
	@devec sb2 = sum(sqr(b), 1)

	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			r[i,j] = (-2) * r[i] + sa2[i] + sb2[j]
		end
	end
end

function pairwise!(r::Array, dist::SqEuclidean, a::Matrix)
	A_mul_Bt(r, a, a)
	@devec sa2 = sum(sqr(a), 1)

	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = (-2) * r[i] + sa2[i] + sa2[j]
		end
		r[j,j] = 0
		for i = j+1 : n
			r[i,j] = (-2) * r[i] + sa2[i] + sa2[j]
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

function pairwise!(r::Array, dist::Euclidean, a::Matrix, b::Matrix)
	pairwise!(r, SqEuclidean(), a, b)
	@devec r[:] = sqrt(max(r, 0))
end

function pairwise!(r::Array, dist::Euclidean, a::Matrix)
	pairwise!(r, SqEuclidean(), a, b)
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

function pairwise!(r::Array, dist::Cityblock, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			@devec r[i,j] = sum(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::Array, dist::Cityblock, a::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			r[i,j] = sum(abs(a[:,i] - a[:,j]))
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
	@devec r[:] = max(abs(a - b), 1)
end

function colwise!(r::Array, dist::Chebyshev, a::Vector, b::Matrix)
	for j = 1 : size(b, 2)
		@devec r[j] = max(abs(a - b[:,j]))
	end
end

function pairwise!(r::Array, dist::Chebyshev, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : m
			@devec r[i,j] = max(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::Array, dist::Chebyshev, a::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			r[i,j] = max(abs(a[:,i] - a[:,j]))
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

function pairwise!(r::Array, dist::Minkowski, a::Matrix, b::Matrix)
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

function pairwise!(r::Array, dist::Minkowski, a::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec t = sum(abs(a[:,i] - b[:,j]) .^ p)
			r[i,j] = t ^ inv_p
		end
	end
end



end # module end
	
	