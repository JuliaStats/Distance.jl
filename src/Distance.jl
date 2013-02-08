module Distance
	
using Devectorize

export 
	# generic types/functions
	GeneralizedMetric, 
	Metric,
	
	# distance classes
	Euclidean,
	SqEuclidean,
	Cityblock,
	Chebyshev,
	Minkowski,
	
	Hamming,
	CosineDist,
	CorrDist,
	ChiSqDist,
	KLDivergence,
	JSDivergence,
	
	WeightedEuclidean,
	WeightedSqEuclidean,
	WeightedCityblock,
	WeightedMinkowski,
	WeightedHamming,
	SqMahalanobis,
	Mahalanobis,
	
	# convenient functions
	euclidean,
	sqeuclidean,
	cityblock,
	chebyshev,
	minkowski,
	mahalanobis,
	
	hamming,
	cosine_dist,
	corr_dist,
	chisq_dist,
	kl_divergence,
	js_divergence,
	
	weighted_euclidean,
	weighted_sqeuclidean,
	weighted_cityblock,
	weighted_minkowski,
	weighted_hamming,
	sqmahalanobis,
	mahalanobis,

	# generic functions 
	result_type,
	colwise,
	pairwise,
	evaluate,
	
	# other convenient functions
	At_Q_B, At_Q_A


include("at_q_b.jl")

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
	
type WeightedEuclidean{T<:FloatingPoint} <: Metric 
	weights::Vector{T}
end
		
type WeightedSqEuclidean{T<:FloatingPoint} <: SemiMetric 
	weights::Vector{T}
end

type WeightedCityblock{T<:FloatingPoint} <: Metric 
	weights::Vector{T}
end
	
type WeightedMinkowski{T<:FloatingPoint} <: Metric 	
	weights::Vector{T}
	p::Real
end

type WeightedHamming{T<:FloatingPoint} <: Metric 
	weights::Vector{T}
end

type Mahalanobis{T} <: Metric 
	qmat::Matrix{T}
end

type SqMahalanobis{T} <: SemiMetric
	qmat::Matrix{T}
end


###########################################################
#
#	result types
#
###########################################################

result_type(::PreMetric, T1::Type, T2::Type) = promote_type(T1, T2)
result_type(::Hamming, T1::Type, T2::Type) = Int

result_type{T}(::WeightedEuclidean{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedSqEuclidean{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedCityblock{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedMinkowski{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedHamming{T}, T1::Type, T2::Type) = T

result_type{T}(::Mahalanobis{T}, T1::Type, T2::Type) = T
result_type{T}(::SqMahalanobis{T}, T1::Type, T2::Type) = T


###########################################################
#
#	helper functions for dimension checking
#
###########################################################

function get_common_ncols(a::Matrix, b::Matrix)
	na = size(a, 2)
	nb = size(b, 2)
	if na != nb
		throw(ArgumentError("The number of columns in a and b must match."))
	end
	return na
end

# the following functions are supposed to be used in inplace-functions
# and they only apply to cases where vector dimensions are the same

function get_colwise_dims(r::Array, a::Matrix, b::Matrix)
	if !(size(a) == size(b))
		throw(ArgumentError("The sizes of a and b must match."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_colwise_dims(r::Array, a::Vector, b::Matrix)
	if length(a) != size(b, 1)
		throw(ArgumentError("The length of a must match the number of rows in b."))
	end
	if length(r) != size(b, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(b)
end

function get_colwise_dims(r::Array, a::Matrix, b::Vector)
	if !(size(a, 1) == length(b))
		throw(ArgumentError("The length of b must match the number of rows in a."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_pairwise_dims(r::Matrix, a::Matrix, b::Matrix)
	ma, na = size(a)
	mb, nb = size(b)
	if ma != mb
		throw(ArgumentError("The numbers of rows in a and b must match."))
	end
	if !(size(r) == (na, nb))
		throw(ArgumentError("Incorrect size of r."))
	end
	return (ma, na, nb)
end

function get_pairwise_dims(r::Matrix, a::Matrix)
	m, n = size(a)
	if !(size(r) == (n, n))
		throw(ArgumentError("Incorrect size of r."))
	end
	return (m, n)
end


# for weighted metrics

function get_colwise_dims(d::Int, r::Array, a::Matrix, b::Matrix)
	if !(size(a, 1) == size(b, 1) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_colwise_dims(d::Int, r::Array, a::Vector, b::Matrix)
	if !(length(a) == size(b, 1) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if length(r) != size(b, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(b)
end

function get_colwise_dims(d::Int, r::Array, a::Matrix, b::Vector)
	if !(size(a, 1) == length(b) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_pairwise_dims(d::Int, r::Matrix, a::Matrix, b::Matrix)
	na = size(a, 2)
	nb = size(b, 2)
	if !(size(a, 1) == size(b, 1) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if !(size(r) == (na, nb))
		throw(ArgumentError("Incorrect size of r."))
	end
	return (d, na, nb)
end

function get_pairwise_dims(d::Int, r::Matrix, a::Matrix)
	n = size(a, 2)
	if !(size(a, 1) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if !(size(r) == (n, n))
		throw(ArgumentError("Incorrect size of r."))
	end
	return (d, n)
end



###########################################################
#
#	Generic colwise and pairwise evaluation
#
###########################################################

# function colwise!(r::Array, metric::PreMetric, a::Vector, b::Matrix)
# 	n = size(b, 2)
# 	if length(r) != n
# 		throw(ArgumentError("Incorrect size of r."))
# 	end
# 	for j = 1 : n
# 		r[j] = evaluate(metric, a, b[:,j])
# 	end
# end
# 
# function colwise!(r::Array, metric::PreMetric, a::Matrix, b::Vector)
# 	n = size(a, 2)
# 	if length(r) != n
# 		throw(ArgumentError("Incorrect size of r."))
# 	end
# 	for j = 1 : n
# 		r[j] = evaluate(metric, a[:,j], b)
# 	end
# end
# 
# function colwise!(r::Array, metric::PreMetric, a::Matrix, b::Matrix)
# 	n = get_common_ncols(a, b)
# 	if length(r) != n
# 		throw(ArgumentError("Incorrect size of r."))
# 	end
# 	for j = 1 : n
# 		r[j] = evaluate(metric, a[:,j], b[:,j])
# 	end
# end

function colwise!(r::Array, metric::SemiMetric, a::Matrix, b::Vector)
	colwise!(r, metric, b, a)
end

function colwise(metric::PreMetric, a::Matrix, b::Matrix)
	n = get_common_ncols(a, b)
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


# function pairwise!(r::Matrix, metric::PreMetric, a::Matrix, b::Matrix)
# 	na = size(a, 2)
# 	nb = size(b, 2)
# 	if !(size(r) == (na, nb))
# 		throw(ArgumentError("Incorrect size of r."))
# 	end
# 	for j = 1 : size(b, 2)
# 		bj = b[:,j]
# 		for i = 1 : size(a, 2)
# 			r[i,j] = evaluate(metric, a[:,i], bj)
# 		end
# 	end
# end

function pairwise!(r::Matrix, metric::PreMetric, a::Matrix)
	pairwise!(r, metric, a, a)
end


# faster evaluation by leveraging the properties of semi-metrics
# function pairwise!(r::Matrix, metric::SemiMetric, a::Matrix)
# 	n = size(a, 2)
# 	if !(size(r) == (n, n))
# 		throw(ArgumentError("Incorrect size of r."))
# 	end
# 	for j = 1 : n
# 		for i = 1 : j-1
# 			a[i,j] = a[j,i]
# 		end
# 		a[j,j] = 0
# 		bj = b[:,j]
# 		for i = j+1 : n
# 			a[i,j] = evaluate(metric, a[:,i], bj)
# 		end
# 	end
# end

function pairwise(metric::PreMetric, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), (m, n))
	pairwise!(r, metric, a, b)
	return r
end

function pairwise(metric::PreMetric, a::Matrix)
	n = size(a, 2)
	r = Array(result_type(metric, eltype(a), eltype(a)), (n, n))
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
	get_colwise_dims(r, a, b)
	@devec r[:] = sum(sqr(a - b), 1)
end

function colwise!(r::Array, dist::SqEuclidean, a::Vector, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a - b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::SqEuclidean, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	At_mul_B(r, a, b)
	@devec sa2 = sum(sqr(a), 1)
	@devec sb2 = sum(sqr(b), 1)
	for j = 1 : nb
		for i = 1 : na
			r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
		end
	end
end

function pairwise!(r::Matrix, dist::SqEuclidean, a::Matrix)
	m, n = get_pairwise_dims(r, a)
	At_mul_B(r, a, a)
	@devec sa2 = sum(sqr(a), 1)
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
	get_colwise_dims(r, a, b)
	@devec r[:] = sqrt(sum(sqr(a - b), 1))
end

function colwise!(r::Array, dist::Euclidean, a::Vector, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a - b[:,j]))
		r[j] = sqrt(r[j])
	end
end

function pairwise!(r::Matrix, dist::Euclidean, a::Matrix, b::Matrix)
	get_pairwise_dims(r, a, b)
	pairwise!(r, SqEuclidean(), a, b)
	@devec r[:] = sqrt(max(r, 0))
end

function pairwise!(r::Matrix, dist::Euclidean, a::Matrix)
	get_pairwise_dims(r, a)
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
	get_colwise_dims(r, a, b)
	@devec r[:] = sum(abs(a - b), 1)
end

function colwise!(r::Array, dist::Cityblock, a::Vector, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	for j = 1 : n
		@devec r[j] = sum(abs(a - b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::Cityblock, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = sum(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::Matrix, dist::Cityblock, a::Matrix)
	m, n = get_pairwise_dims(r, a)
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
	get_colwise_dims(r, a, b)
	@devec r[:] = max(abs(a - b), (), 1)
end

function colwise!(r::Array, dist::Chebyshev, a::Vector, b::Matrix)
	get_colwise_dims(r, a, b)
	for j = 1 : size(b, 2)
		@devec r[j] = max(abs(a - b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::Chebyshev, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = max(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::Matrix, dist::Chebyshev, a::Matrix)
	m, n = get_pairwise_dims(r, a)
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
	get_colwise_dims(r, a, b)
	p = dist.p
	inv_p = 1 / p
	@devec r[:] = sum(abs(a - b) .^ p, 1) .^ inv_p
end

function colwise!(r::Array, dist::Minkowski, a::Vector, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		@devec r[j] = sum(abs(a - b[:,j]) .^ p)
		r[j] = r[j] ^ inv_p
	end
end

function pairwise!(r::Matrix, dist::Minkowski, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : nb
		for i = 1 : na
			@devec t = sum(abs(a[:,i] - b[:,j]) .^ p)
			r[i,j] = t ^ inv_p
		end
	end
end

function pairwise!(r::Matrix, dist::Minkowski, a::Matrix)
	m, n = get_pairwise_dims(r, a)
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
	m, n = get_colwise_dims(r, a, b)
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
	m, n = get_colwise_dims(r, a, b)
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
	m, na, nb = get_pairwise_dims(r, a, b)
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
	m, n = get_pairwise_dims(r, a)
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
	get_colwise_dims(r, a, b)
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
	get_colwise_dims(r, a, b)
	@devec begin
		ra = sqrt(sum(sqr(a)))
		rb = sum(sqr(b), 1)
		rb[:] = sqrt(rb)
	end
	ab = At_mul_B(b, a)
	@devec r[:] = max(1 - ab ./ (ra .* rb), 0)
end

function pairwise!(r::Matrix, dist::CosineDist, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	At_mul_B(r, a, b)
	@devec begin
		ra = sum(sqr(a), 1)
		rb = sum(sqr(b), 1)
		ra[:] = sqrt(ra)
		rb[:] = sqrt(rb)
	end
	for j = 1 : nb
		for i = 1 : na
			r[i,j] = max(1 - r[i,j] / (ra[i] * rb[j]), 0)
		end
	end
end

function pairwise!(r::Matrix, dist::CosineDist, a::Matrix)
	m, n = get_pairwise_dims(r, a)
	At_mul_B(r, a, a)
	@devec begin
		ra = sum(sqr(a), 1)
		ra[:] = sqrt(ra)
	end
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
	get_colwise_dims(r, a, b)
	@devec r[:] = sum(sqr(a - b) ./ (a + b), 1)
end

function colwise!(r::Array, dist::ChiSqDist, a::Vector, b::Matrix)
	get_colwise_dims(r, a, b)
	for j = 1 : size(b, 2)
		@devec r[j] = sum(sqr(a - b[:,j]) ./ (a + b[:,j]))
	end
end

function pairwise!(r::Matrix, dist::ChiSqDist, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = sum(sqr(a[:,i] - b[:,j]) ./ (a[:,i] + b[:,j]))
		end
	end
end

function pairwise!(r::Matrix, dist::ChiSqDist, a::Matrix)
	m, n = get_pairwise_dims(r, a)
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


# KL divergence

function evaluate(dist::KLDivergence, a::Vector, b::Vector)
	r = zero(promote_type(eltype(a), eltype(b)))
	n = length(a)
	if n != length(b)
		throw(ArgumentError("The lengths of a and b must match."))
	end
	for i = 1 : n
		if a[i] > 0
			r += a[i] * log(a[i] / b[i])
		end
	end
	return r
end

kl_divergence(a::Vector, b::Vector) = evaluate(KLDivergence(), a, b)

function colwise!(r::Array, dist::KLDivergence, a::Matrix, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			if a[i,j] > 0
				s += a[i,j] * log(a[i,j] / b[i,j])
			end
		end
		r[j] = s
	end
end

function colwise!(r::Array, dist::KLDivergence, a::Vector, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			if a[i] > 0
				s += a[i] * log(a[i] / b[i,j])
			end
		end
		r[j] = s
	end
end


function colwise!(r::Array, dist::KLDivergence, a::Matrix, b::Vector)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			if a[i,j] > 0
				s += a[i,j] * log(a[i,j] / b[i])
			end
		end
		r[j] = s
	end
end


function pairwise!(r::Matrix, dist::KLDivergence, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : nb
		for i = 1 : na
			s = zero(T)
			for k = 1 : m
				if a[k,i] > 0
					s += a[k,i] * log(a[k,i] / b[k,j])		
				end
			end
			r[i,j] = s
		end
	end
end

function pairwise!(r::Matrix, dist::KLDivergence, a::Matrix)
	pairwise!(r, dist, a, a)  # K-L divergence is not symmetric
end


# JS divergence

function evaluate(dist::JSDivergence, a::Vector, b::Vector)
	r = zero(promote_type(eltype(a), eltype(b)))
	n = length(a)
	if n != length(b)
		throw(ArgumentError("The lengths of a and b must match."))
	end
	for i = 1 : n
		u = (a[i] + b[i]) / 2
		ta = a[i] > 0 ? a[i] * log(a[i]) / 2 : 0
		tb = b[i] > 0 ? b[i] * log(b[i]) / 2: 0
		tu = u > 0 ? u * log(u) : 0
		r += (ta + tb - tu)
	end
	return r
end

js_divergence(a::Vector, b::Vector) = evaluate(JSDivergence(), a, b)

function colwise!(r::Array, dist::JSDivergence, a::Matrix, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			u = (a[i,j] + b[i,j]) / 2
			ta = a[i,j] > 0 ? a[i,j] * log(a[i,j]) / 2 : 0
			tb = b[i,j] > 0 ? b[i,j] * log(b[i,j]) / 2: 0
			tu = u > 0 ? u * log(u) : 0
			s += (ta + tb - tu)
		end
		r[j] = s
	end
end

function colwise!(r::Array, dist::JSDivergence, a::Vector, b::Matrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			u = (a[i] + b[i,j]) / 2
			ta = a[i] > 0 ? a[i] * log(a[i]) / 2 : 0
			tb = b[i,j] > 0 ? b[i,j] * log(b[i,j]) / 2: 0
			tu = u > 0 ? u * log(u) : 0
			s += (ta + tb - tu)
		end
		r[j] = s
	end
end

function pairwise!(r::Matrix, dist::JSDivergence, a::Matrix, b::Matrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : nb
		for i = 1 : na
			s = zero(T)
			for k = 1 : m
				if a[k,i] > 0
					u = (a[k,i] + b[k,j]) / 2
					ta = a[k,i] > 0 ? a[k,i] * log(a[k,i]) / 2 : 0
					tb = b[k,j] > 0 ? b[k,j] * log(b[k,j]) / 2: 0
					tu = u > 0 ? u * log(u) : 0
					s += (ta + tb - tu)	
				end
			end
			r[i,j] = s
		end
	end
end

function pairwise!(r::Matrix, dist::JSDivergence, a::Matrix)
	m, n = get_pairwise_dims(r, a)
	T = eltype(a)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			s = zero(T)
			for k = 1 : m
				if a[k,i] > 0
					u = (a[k,i] + a[k,j]) / 2
					ta = a[k,i] > 0 ? a[k,i] * log(a[k,i]) / 2 : 0
					tb = a[k,j] > 0 ? a[k,j] * log(a[k,j]) / 2 : 0
					tu = u > 0 ? u * log(u) : 0
					s += (ta + tb - tu)	
				end
			end
			r[i,j] = s
		end
	end
end


# Weighted squared Euclidean

function evaluate{T<:FloatingPoint}(dist::WeightedSqEuclidean{T}, a::Vector, b::Vector)
	w = dist.weights
	@devec r = sum(sqr(a - b) .* w)
	return r
end

weighted_sqeuclidean(a::Vector, b::Vector, w::Vector) = evaluate(WeightedSqEuclidean(w), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedSqEuclidean{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a[:,j] - b[:,j]) .* w)
	end
end

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedSqEuclidean{T}, a::Vector, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a - b[:,j]) .* w)
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedSqEuclidean{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, na, nb = get_pairwise_dims(length(w), r, a, b)
	
	sa2 = Array(T, na)
	sb2 = Array(T, nb)
	for i = 1 : na
		@devec sa2[i] = sum(sqr(a[:,i]) .* w)		
	end
	for j = 1 : nb
		@devec sb2[j] = sum(sqr(b[:,j]) .* w)
	end
		
	At_Q_B!(r, w, a, b)
	for j = 1 : nb
		for i = 1 : na
			r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
		end
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedSqEuclidean{T}, a::Matrix)
	w = dist.weights
	m, n = get_pairwise_dims(length(w), r, a)
	
	sa2 = Array(T, n)
	for i = 1 : n
		@devec sa2[i] = sum(sqr(a[:,i]) .* w)		
	end
	
	At_Q_A!(r, w, a)
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


# Weighted Euclidean

function evaluate{T<:FloatingPoint}(dist::WeightedEuclidean{T}, a::Vector, b::Vector)
	sqrt(evaluate(WeightedSqEuclidean(dist.weights), a, b))
end

weighted_euclidean(a::Vector, b::Vector, w::Vector) = evaluate(WeightedEuclidean(w), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedEuclidean{T}, a::Matrix, b::Matrix)
	colwise!(r, WeightedSqEuclidean(dist.weights), a, b)
	@devec r[:] = sqrt(r)
end

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedEuclidean{T}, a::Vector, b::Matrix)
	colwise!(r, WeightedSqEuclidean(dist.weights), a, b)
	@devec r[:] = sqrt(r)
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedEuclidean{T}, a::Matrix, b::Matrix)
	pairwise!(r, WeightedSqEuclidean(dist.weights), a, b)
	@devec r[:] = sqrt(max(r, 0))
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedEuclidean{T}, a::Matrix)
	pairwise!(r, WeightedSqEuclidean(dist.weights), a)
	@devec r[:] = sqrt(max(r, 0))
end

# Weighted Cityblock

function evaluate{T<:FloatingPoint}(dist::WeightedCityblock{T}, a::Vector, b::Vector)
	w = dist.weights
	@devec r = sum(abs(a - b) .* w)
	return r
end

weighted_cityblock(a::Vector, b::Vector, w::Vector) = evaluate(WeightedCityblock(w), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedCityblock{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		@devec r[j] = sum(abs(a[:,j] - b[:,j]) .* w)
	end
end

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedCityblock{T}, a::Vector, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)	
	for j = 1 : n
		@devec r[j] = sum(abs(a - b[:,j]) .* w)
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedCityblock{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, na, nb = get_pairwise_dims(length(w), r, a, b)	
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = sum(abs(a[:,i] - b[:,j]) .* w)
		end
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedCityblock{T}, a::Matrix)
	w = dist.weights
	m, n = get_pairwise_dims(length(w), r, a)	
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec r[i,j] = sum(abs(a[:,i] - a[:,j]) .* w)
		end
	end
end


# WeightedMinkowski

function evaluate{T<:FloatingPoint}(dist::WeightedMinkowski{T}, a::Vector, b::Vector)
	p = dist.p
	w = dist.weights
	@devec r = sum((abs(a - b) .^ p) .* w)
	return r ^ (1 / p)
end

weighted_minkowski(a::Vector, b::Vector, w::Vector, p::Real) = evaluate(WeightedMinkowski(w, p), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedMinkowski{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	p = dist.p
	inv_p = 1 / p
	for j = 1 : n
		@devec s = sum((abs(a[:,j] - b[:,j]) .^ p) .* w)
		r[j] = s ^ inv_p
	end
end

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedMinkowski{T}, a::Vector, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		@devec s = sum((abs(a - b[:,j]) .^ p) .* w)
		r[j] = s ^ inv_p
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedMinkowski{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, na, nb = get_pairwise_dims(length(w), r, a, b)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : nb
		for i = 1 : na
			@devec t = sum((abs(a[:,i] - b[:,j]) .^ p) .* w)
			r[i,j] = t ^ inv_p
		end
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedMinkowski{T}, a::Matrix)
	w = dist.weights
	m, n = get_pairwise_dims(r, a)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			@devec t = sum((abs(a[:,i] - a[:,j]) .^ p) .* w)
			r[i,j] = t ^ inv_p
		end
	end
end


# WeightedHamming

function evaluate{T<:FloatingPoint}(dist::WeightedHamming{T}, a::Vector, b::Vector)
	w = dist.weights
	sum((a .!= b) .* w)
end

weighted_hamming(a::Vector, b::Vector, w::Vector) = evaluate(WeightedHamming(w), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedHamming{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		d = zero(T)
		for i = 1 : m
			if (a[i,j] != b[i,j]) 
				d += w[i]
			end
		end
		r[j] = d
	end
end

function colwise!{T<:FloatingPoint}(r::Array, dist::WeightedHamming{T}, a::Vector, b::Matrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		d = zero(T)
		for i = 1 : m
			if (a[i] != b[i,j]) 
				d += w[i]
			end
		end
		r[j] = d
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedHamming{T}, a::Matrix, b::Matrix)
	w = dist.weights
	m, na, nb = get_pairwise_dims(length(w), r, a, b)
	for j = 1 : nb
		for i = 1 : na
			d = zero(T)
			for k = 1 : m
				if a[k,i] != b[k,j]
					d += w[k]
				end	
			end
			r[i,j] = d
		end
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::WeightedHamming{T}, a::Matrix)
	w = dist.weights
	m, n = get_pairwise_dims(r, a)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			d = zero(T)
			for k = 1 : m
				if a[k,i] != a[k,j]
					d += w[k]
				end
			end
			r[i,j] = d
		end
	end
end


# SqMahalanobis

function evaluate{T<:FloatingPoint}(dist::SqMahalanobis{T}, a::Vector, b::Vector)
	Q = dist.qmat
	z = a - b
	return dot(z, Q * z)
end

sqmahalanobis(a::Vector, b::Vector, Q::Matrix) = evaluate(SqMahalanobis(Q), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::SqMahalanobis{T}, a::Matrix, b::Matrix)
	Q = dist.qmat
	m, n = get_colwise_dims(size(Q, 1), r, a, b)
	z = a - b
	Qz = Q * z
	@devec r[:] = sum(Qz .* z, 1)
end

function colwise!{T<:FloatingPoint}(r::Array, dist::SqMahalanobis{T}, a::Vector, b::Matrix)
	Q = dist.qmat
	m, n = get_colwise_dims(size(Q, 1), r, a, b)
	z = Array(T, (m, n))
	for j = 1 : n
		@devec z[:,j] = a - b[:,j]
	end
	Qz = Q * z
	@devec r[:] = sum(Qz .* z, 1)
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::SqMahalanobis{T}, a::Matrix, b::Matrix)
	Q = dist.qmat
	m, na, nb = get_pairwise_dims(size(Q, 1), r, a, b)
	
	Qa = Q * a
	Qb = Q * b
	@devec sa2 = sum(a .* Qa, 1)
	@devec sb2 = sum(b .* Qb, 1)
	At_mul_B(r, a, Qb)
	
	for j = 1 : nb
		for i = 1 : na
			r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
		end
	end
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::SqMahalanobis{T}, a::Matrix)
	Q = dist.qmat
	m, n = get_pairwise_dims(size(Q, 1), r, a)
	
	Qa = Q * a
	@devec sa2 = sum(a .* Qa, 1)
	At_mul_B(r, a, Qa)

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


# Mahalanobis

function evaluate{T<:FloatingPoint}(dist::Mahalanobis{T}, a::Vector, b::Vector)
	sqrt(evaluate(SqMahalanobis(dist.qmat), a, b))
end

mahalanobis(a::Vector, b::Vector, Q::Matrix) = evaluate(Mahalanobis(Q), a, b)

function colwise!{T<:FloatingPoint}(r::Array, dist::Mahalanobis{T}, a::Matrix, b::Matrix)
	colwise!(r, SqMahalanobis(dist.qmat), a, b)
	@devec r[:] = sqrt(r)
end

function colwise!{T<:FloatingPoint}(r::Array, dist::Mahalanobis{T}, a::Vector, b::Matrix)
	colwise!(r, SqMahalanobis(dist.qmat), a, b)
	@devec r[:] = sqrt(r)
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::Mahalanobis{T}, a::Matrix, b::Matrix)
	pairwise!(r, SqMahalanobis(dist.qmat), a, b)
	@devec r[:] = sqrt(r)
end

function pairwise!{T<:FloatingPoint}(r::Matrix, dist::Mahalanobis{T}, a::Matrix)
	pairwise!(r, SqMahalanobis(dist.qmat), a)
	@devec r[:] = sqrt(r)
end


end # module end
	
	