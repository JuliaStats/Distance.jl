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

function get_common_ncols(a::AbstractMatrix, b::AbstractMatrix)
	na = size(a, 2)
	nb = size(b, 2)
	if na != nb
		throw(ArgumentError("The number of columns in a and b must match."))
	end
	return na
end

# the following functions are supposed to be used in inplace-functions
# and they only apply to cases where vector dimensions are the same

function get_colwise_dims(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
	if !(size(a) == size(b))
		throw(ArgumentError("The sizes of a and b must match."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_colwise_dims(r::AbstractArray, a::AbstractVector, b::AbstractMatrix)
	if length(a) != size(b, 1)
		throw(ArgumentError("The length of a must match the number of rows in b."))
	end
	if length(r) != size(b, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(b)
end

function get_colwise_dims(r::AbstractArray, a::AbstractMatrix, b::AbstractVector)
	if !(size(a, 1) == length(b))
		throw(ArgumentError("The length of b must match the number of rows in a."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_pairwise_dims(r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
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

function get_pairwise_dims(r::AbstractMatrix, a::AbstractMatrix)
	m, n = size(a)
	if !(size(r) == (n, n))
		throw(ArgumentError("Incorrect size of r."))
	end
	return (m, n)
end


# for weighted metrics

function get_colwise_dims(d::Int, r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
	if !(size(a, 1) == size(b, 1) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_colwise_dims(d::Int, r::AbstractArray, a::AbstractVector, b::AbstractMatrix)
	if !(length(a) == size(b, 1) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if length(r) != size(b, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(b)
end

function get_colwise_dims(d::Int, r::AbstractArray, a::AbstractMatrix, b::AbstractVector)
	if !(size(a, 1) == length(b) == d)
		throw(ArgumentError("Incorrect vector dimensions."))
	end
	if length(r) != size(a, 2)
		throw(ArgumentError("Incorrect size of r."))
	end
	return size(a)
end

function get_pairwise_dims(d::Int, r::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
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

function get_pairwise_dims(d::Int, r::AbstractMatrix, a::AbstractMatrix)
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

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
	n = size(b, 2)
	if length(r) != n
		throw(ArgumentError("Incorrect size of r."))
	end
	for j = 1 : n
		r[j] = evaluate(metric, a, b[:,j])
	end
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
	n = size(a, 2)
	if length(r) != n
		throw(ArgumentError("Incorrect size of r."))
	end
	for j = 1 : n
		r[j] = evaluate(metric, a[:,j], b)
	end
end

function colwise!(r::AbstractArray, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
	n = get_common_ncols(a, b)
	if length(r) != n
		throw(ArgumentError("Incorrect size of r."))
	end
	for j = 1 : n
		r[j] = evaluate(metric, a[:,j], b[:,j])
	end
end

function colwise!(r::AbstractArray, metric::SemiMetric, a::AbstractMatrix, b::AbstractVector)
	colwise!(r, metric, b, a)
end

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
	n = get_common_ncols(a, b)
	r = Array(result_type(metric, eltype(a), eltype(b)), n)
	colwise!(r, metric, a, b)
	return r
end

function colwise(metric::PreMetric, a::AbstractVector, b::AbstractMatrix)
	n = size(b, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), n)
	colwise!(r, metric, a, b)
	return r
end

function colwise(metric::PreMetric, a::AbstractMatrix, b::AbstractVector)
	n = size(a, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), n)
	colwise!(r, metric, a, b)
	return r
end


function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
	na = size(a, 2)
	nb = size(b, 2)
	if !(size(r) == (na, nb))
		throw(ArgumentError("Incorrect size of r."))
	end
	for j = 1 : size(b, 2)
		bj = b[:,j]
		for i = 1 : size(a, 2)
			r[i,j] = evaluate(metric, a[:,i], bj)
		end
	end
end

function pairwise!(r::AbstractMatrix, metric::PreMetric, a::AbstractMatrix)
	pairwise!(r, metric, a, a)
end


# faster evaluation by leveraging the properties of semi-metrics
function pairwise!(r::AbstractMatrix, metric::SemiMetric, a::AbstractMatrix)
	n = size(a, 2)
	if !(size(r) == (n, n))
		throw(ArgumentError("Incorrect size of r."))
	end
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

function pairwise(metric::PreMetric, a::AbstractMatrix, b::AbstractMatrix)
	m = size(a, 2)
	n = size(b, 2)
	r = Array(result_type(metric, eltype(a), eltype(b)), (m, n))
	pairwise!(r, metric, a, b)
	return r
end

function pairwise(metric::PreMetric, a::AbstractMatrix)
	n = size(a, 2)
	r = Array(result_type(metric, eltype(a), eltype(a)), (n, n))
	pairwise!(r, metric, a)
	return r
end

function pairwise(metric::SemiMetric, a::AbstractMatrix)
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

function evaluate(dist::SqEuclidean, a::AbstractVector, b::AbstractVector)
	@devec r = sum(sqr(a - b))
	return r
end

sqeuclidean(a::AbstractVector, b::AbstractVector) = evaluate(SqEuclidean(), a, b)

function colwise!(r::AbstractArray, dist::SqEuclidean, a::AbstractMatrix, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	@devec r[:] = sum(sqr(a - b), 1)
end

function colwise!(r::AbstractArray, dist::SqEuclidean, a::AbstractVector, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a - b[:,j]))
	end
end

function pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!(r::AbstractMatrix, dist::SqEuclidean, a::AbstractMatrix)
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

function evaluate(dist::Euclidean, a::AbstractVector, b::AbstractVector)
	@devec r = sum(sqr(a - b))
	return sqrt(r)
end

euclidean(a::AbstractVector, b::AbstractVector) = evaluate(Euclidean(), a, b)

function colwise!(r::AbstractArray, dist::Euclidean, a::AbstractMatrix, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	@devec r[:] = sqrt(sum(sqr(a - b), 1))
end

function colwise!(r::AbstractArray, dist::Euclidean, a::AbstractVector, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a - b[:,j]))
		r[j] = sqrt(r[j])
	end
end

function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	At_mul_B(r, a, b)
	@devec sa2 = sum(sqr(a), 1)
	@devec sb2 = sum(sqr(b), 1)
	for j = 1 : nb
		for i = 1 : na
			v = sa2[i] + sb2[j] - 2 * r[i,j]
			r[i,j] = sqrt(max(v, 0))
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::Euclidean, a::AbstractMatrix)
	m, n = get_pairwise_dims(r, a)
	At_mul_B(r, a, a)
	@devec sa2 = sum(sqr(a), 1)
	for j = 1 : n
		for i = 1 : j-1
			r[i,j] = r[j,i]
		end
		r[j,j] = 0
		for i = j+1 : n
			v = sa2[i] + sa2[j] - 2 * r[i,j]
			r[i,j] = sqrt(max(v, 0))
		end
	end
end


# Cityblock

function evaluate(dist::Cityblock, a::AbstractVector, b::AbstractVector)
	@devec r = sum(abs(a - b))
	return r
end

cityblock(a::AbstractVector, b::AbstractVector) = evaluate(Cityblock(), a, b)

function colwise!(r::AbstractArray, dist::Cityblock, a::AbstractMatrix, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	@devec r[:] = sum(abs(a - b), 1)
end

function colwise!(r::AbstractArray, dist::Cityblock, a::AbstractVector, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	for j = 1 : n
		@devec r[j] = sum(abs(a - b[:,j]))
	end
end

function pairwise!(r::AbstractMatrix, dist::Cityblock, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = sum(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::Cityblock, a::AbstractMatrix)
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

function evaluate(dist::Chebyshev, a::AbstractVector, b::AbstractVector)
	@devec r = max(abs(a - b))
	return r
end

chebyshev(a::AbstractVector, b::AbstractVector) = evaluate(Chebyshev(), a, b)

function colwise!(r::AbstractArray, dist::Chebyshev, a::AbstractMatrix, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	@devec r[:] = max(abs(a - b), (), 1)
end

function colwise!(r::AbstractArray, dist::Chebyshev, a::AbstractVector, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	for j = 1 : size(b, 2)
		@devec r[j] = max(abs(a - b[:,j]))
	end
end

function pairwise!(r::AbstractMatrix, dist::Chebyshev, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = max(abs(a[:,i] - b[:,j]))
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::Chebyshev, a::AbstractMatrix)
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

function evaluate(dist::Minkowski, a::AbstractVector, b::AbstractVector)
	p = dist.p
	@devec r = sum(abs(a - b) .^ p)
	return r ^ (1 / p)
end

minkowski(a::AbstractVector, b::AbstractVector, p::Real) = evaluate(Minkowski(p), a, b)

function colwise!(r::AbstractArray, dist::Minkowski, a::AbstractMatrix, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	p = dist.p
	inv_p = 1 / p
	@devec r[:] = sum(abs(a - b) .^ p, 1) .^ inv_p
end

function colwise!(r::AbstractArray, dist::Minkowski, a::AbstractVector, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		@devec r[j] = sum(abs(a - b[:,j]) .^ p)
		r[j] = r[j] ^ inv_p
	end
end

function pairwise!(r::AbstractMatrix, dist::Minkowski, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	p = dist.p
	inv_p = 1 / p
	
	for j = 1 : nb
		for i = 1 : na
			@devec t = sum(abs(a[:,i] - b[:,j]) .^ p)
			r[i,j] = t .^ inv_p
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::Minkowski, a::AbstractMatrix)
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

function evaluate(dist::Hamming, a::AbstractVector, b::AbstractVector)
	n = length(a)
	if n != length(b)
		throw(ArgumentError("The length of a and b must match."))
	end
	
	r = 0
	for i = 1 : n
		if a[i] != b[i]
			r += 1
		end
	end
	return r
end

hamming(a::AbstractVector, b::AbstractVector) = evaluate(Hamming(), a, b)

function colwise!(r::AbstractArray, dist::Hamming, a::AbstractMatrix, b::AbstractMatrix)
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

function colwise!(r::AbstractArray, dist::Hamming, a::AbstractVector, b::AbstractMatrix)
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

function pairwise!(r::AbstractMatrix, dist::Hamming, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!(r::AbstractMatrix, dist::Hamming, a::AbstractMatrix)
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

function evaluate(dist::CosineDist, a::AbstractVector, b::AbstractVector)
	max(1 - dot(a, b) / (norm(a) * norm(b)), 0)
end

cosine_dist(a::AbstractVector, b::AbstractVector) = evaluate(CosineDist(), a, b)

function colwise!(r::AbstractArray, dist::CosineDist, a::AbstractMatrix, b::AbstractMatrix)
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

function colwise!(r::AbstractArray, dist::CosineDist, a::AbstractVector, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	@devec begin
		ra = sqrt(sum(sqr(a)))
		rb = sum(sqr(b), 1)
		rb[:] = sqrt(rb)
	end
	ab = At_mul_B(b, a)
	@devec r[:] = max(1 - ab ./ (ra .* rb), 0)
end

function pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!(r::AbstractMatrix, dist::CosineDist, a::AbstractMatrix)
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

function evaluate(dist::CorrDist, a::AbstractVector, b::AbstractVector)
	cosine_dist(a - mean(a), b - mean(b))
end

corr_dist(a::AbstractVector, b::AbstractVector) = evaluate(CorrDist(), a, b)

function shift_vecs_forcorr(a::AbstractMatrix)
	@devec am = mean(a, 1)
	r = similar(a)
	for j = 1 : size(a, 2)
		@devec r[:,j] = a[:,j] - am[j]
	end
	return r
end

function colwise!(r::AbstractArray, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
	a_ = shift_vecs_forcorr(a)
	b_ = shift_vecs_forcorr(b)
	colwise!(r, CosineDist(), a_, b_)
end

function colwise!(r::AbstractArray, dist::CorrDist, a::AbstractVector, b::AbstractMatrix)
	a_ = a - mean(a)
	b_ = shift_vecs_forcorr(b)
	colwise!(r, CosineDist(), a_, b_)
end

function pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix, b::AbstractMatrix)
	a_ = shift_vecs_forcorr(a)
	b_ = shift_vecs_forcorr(b)
	pairwise!(r, CosineDist(), a_, b_)
end

function pairwise!(r::AbstractMatrix, dist::CorrDist, a::AbstractMatrix)
	a_ = shift_vecs_forcorr(a)
	pairwise!(r, CosineDist(), a_)
end


# Chi-square distance

function evaluate(dist::ChiSqDist, a::AbstractVector, b::AbstractVector)
	@devec r = sum(sqr(a - b) ./ (a + b))
	return r
end

chisq_dist(a::AbstractVector, b::AbstractVector) = evaluate(ChiSqDist(), a, b)

function colwise!(r::AbstractArray, dist::ChiSqDist, a::AbstractMatrix, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	@devec r[:] = sum(sqr(a - b) ./ (a + b), 1)
end

function colwise!(r::AbstractArray, dist::ChiSqDist, a::AbstractVector, b::AbstractMatrix)
	get_colwise_dims(r, a, b)
	for j = 1 : size(b, 2)
		@devec r[j] = sum(sqr(a - b[:,j]) ./ (a + b[:,j]))
	end
end

function pairwise!(r::AbstractMatrix, dist::ChiSqDist, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	m = size(a, 2)
	n = size(b, 2)
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = sum(sqr(a[:,i] - b[:,j]) ./ (a[:,i] + b[:,j]))
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::ChiSqDist, a::AbstractMatrix)
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

function evaluate(dist::KLDivergence, a::AbstractVector, b::AbstractVector)
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

kl_divergence(a::AbstractVector, b::AbstractVector) = evaluate(KLDivergence(), a, b)

function colwise!(r::AbstractArray, dist::KLDivergence, a::AbstractMatrix, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			aij = a[i,j]
			if aij > 0
				s += aij * log(aij / b[i,j])
			end
		end
		r[j] = s
	end
end

function colwise!(r::AbstractArray, dist::KLDivergence, a::AbstractVector, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			ai = a[i]
			if ai > 0
				s += ai * log(ai / b[i,j])
			end
		end
		r[j] = s
	end
end


function colwise!(r::AbstractArray, dist::KLDivergence, a::AbstractMatrix, b::AbstractVector)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			aij = a[i,j]
			if aij > 0
				s += aij * log(aij / b[i])
			end
		end
		r[j] = s
	end
end


function pairwise!(r::AbstractMatrix, dist::KLDivergence, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : nb
		for i = 1 : na
			s = zero(T)
			for k = 1 : m
				aki = a[k,i]
				if aki > 0
					s += aki * log(aki / b[k,j])		
				end
			end
			r[i,j] = s
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::KLDivergence, a::AbstractMatrix)
	pairwise!(r, dist, a, a)  # K-L divergence is not symmetric
end


# JS divergence

function evaluate(dist::JSDivergence, a::AbstractVector, b::AbstractVector)
	r = zero(promote_type(eltype(a), eltype(b)))
	n = length(a)
	if n != length(b)
		throw(ArgumentError("The lengths of a and b must match."))
	end
	for i = 1 : n
		ai = a[i]
		bi = b[i]
		u = (ai + bi) / 2
		ta = ai > 0 ? ai * log(ai) / 2 : 0
		tb = bi > 0 ? bi * log(bi) / 2 : 0
		tu = u > 0 ? u * log(u) : 0
		r += (ta + tb - tu)
	end
	return r
end

js_divergence(a::AbstractVector, b::AbstractVector) = evaluate(JSDivergence(), a, b)

function colwise!(r::AbstractArray, dist::JSDivergence, a::AbstractMatrix, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			aij = a[i,j]
			bij = b[i,j]
			u = (aij + bij) / 2
			ta = aij > 0 ? aij * log(aij) / 2 : 0
			tb = bij > 0 ? bij * log(bij) / 2 : 0
			tu = u > 0 ? u * log(u) : 0
			s += (ta + tb - tu)
		end
		r[j] = s
	end
end

function colwise!(r::AbstractArray, dist::JSDivergence, a::AbstractVector, b::AbstractMatrix)
	m, n = get_colwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : n
		s = zero(T)
		for i = 1 : m
			ai = a[i]
			bij = b[i,j]
			u = (ai + bij) / 2
			ta = ai > 0 ? ai * log(ai) / 2 : 0
			tb = bij > 0 ? bij * log(bij) / 2 : 0
			tu = u > 0 ? u * log(u) : 0
			s += (ta + tb - tu)
		end
		r[j] = s
	end
end

function pairwise!(r::AbstractMatrix, dist::JSDivergence, a::AbstractMatrix, b::AbstractMatrix)
	m, na, nb = get_pairwise_dims(r, a, b)
	T = zero(promote_type(eltype(a), eltype(b)))
	for j = 1 : nb
		for i = 1 : na
			s = zero(T)
			for k = 1 : m
				aki = a[k,i]
				bkj = b[k,j]
				u = (aki + bkj) / 2
				ta = aki > 0 ? aki * log(aki) / 2 : 0
				tb = bkj > 0 ? bkj * log(bkj) / 2: 0
				tu = u > 0 ? u * log(u) : 0
				s += (ta + tb - tu)	
			end
			r[i,j] = s
		end
	end
end

function pairwise!(r::AbstractMatrix, dist::JSDivergence, a::AbstractMatrix)
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
				aki = a[k,i]
				akj = a[k,j]
				u = (aki + akj) / 2
				ta = aki > 0 ? aki * log(aki) / 2 : 0
				tb = akj > 0 ? akj * log(akj) / 2 : 0
				tu = u > 0 ? u * log(u) : 0
				s += (ta + tb - tu)	
			end
			r[i,j] = s
		end
	end
end


# Weighted squared Euclidean

function evaluate{T<:FloatingPoint}(dist::WeightedSqEuclidean{T}, a::AbstractVector, b::AbstractVector)
	w = dist.weights
	@devec r = sum(sqr(a - b) .* w)
	return r
end

sqeuclidean(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedSqEuclidean(w), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedSqEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a[:,j] - b[:,j]) .* w)
	end
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedSqEuclidean{T}, a::AbstractVector, b::AbstractMatrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		@devec r[j] = sum(sqr(a - b[:,j]) .* w)
	end
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedSqEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedSqEuclidean{T}, a::AbstractMatrix)
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

function evaluate{T<:FloatingPoint}(dist::WeightedEuclidean{T}, a::AbstractVector, b::AbstractVector)
	sqrt(evaluate(WeightedSqEuclidean(dist.weights), a, b))
end

euclidean(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedEuclidean(w), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
	colwise!(r, WeightedSqEuclidean(dist.weights), a, b)
	@devec r[:] = sqrt(r)
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedEuclidean{T}, a::AbstractVector, b::AbstractMatrix)
	colwise!(r, WeightedSqEuclidean(dist.weights), a, b)
	@devec r[:] = sqrt(r)
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
	pairwise!(r, WeightedSqEuclidean(dist.weights), a, b)
	@devec r[:] = sqrt(max(r, 0))
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedEuclidean{T}, a::AbstractMatrix)
	pairwise!(r, WeightedSqEuclidean(dist.weights), a)
	@devec r[:] = sqrt(max(r, 0))
end

# Weighted Cityblock

function evaluate{T<:FloatingPoint}(dist::WeightedCityblock{T}, a::AbstractVector, b::AbstractVector)
	w = dist.weights
	@devec r = sum(abs(a - b) .* w)
	return r
end

cityblock(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedCityblock(w), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedCityblock{T}, a::AbstractMatrix, b::AbstractMatrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	for j = 1 : n
		@devec r[j] = sum(abs(a[:,j] - b[:,j]) .* w)
	end
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedCityblock{T}, a::AbstractVector, b::AbstractMatrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)	
	for j = 1 : n
		@devec r[j] = sum(abs(a - b[:,j]) .* w)
	end
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedCityblock{T}, a::AbstractMatrix, b::AbstractMatrix)
	w = dist.weights
	m, na, nb = get_pairwise_dims(length(w), r, a, b)	
	for j = 1 : nb
		for i = 1 : na
			@devec r[i,j] = sum(abs(a[:,i] - b[:,j]) .* w)
		end
	end
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedCityblock{T}, a::AbstractMatrix)
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

function evaluate{T<:FloatingPoint}(dist::WeightedMinkowski{T}, a::AbstractVector, b::AbstractVector)
	p = dist.p
	w = dist.weights
	@devec r = sum((abs(a - b) .^ p) .* w)
	return r ^ (1 / p)
end

minkowski(a::AbstractVector, b::AbstractVector, w::AbstractVector, p::Real) = evaluate(WeightedMinkowski(w, p), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedMinkowski{T}, a::AbstractMatrix, b::AbstractMatrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	p = dist.p
	inv_p = 1 / p
	for j = 1 : n
		@devec s = sum((abs(a[:,j] - b[:,j]) .^ p) .* w)
		r[j] = s ^ inv_p
	end
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedMinkowski{T}, a::AbstractVector, b::AbstractMatrix)
	w = dist.weights
	m, n = get_colwise_dims(length(w), r, a, b)
	p = dist.p
	inv_p = 1 / p

	for j = 1 : n
		@devec s = sum((abs(a - b[:,j]) .^ p) .* w)
		r[j] = s ^ inv_p
	end
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedMinkowski{T}, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedMinkowski{T}, a::AbstractMatrix)
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

function evaluate{T<:FloatingPoint}(dist::WeightedHamming{T}, a::AbstractVector, b::AbstractVector)
	n = length(a)
	if n != length(b)
		throw(ArgumentError("The lengths of a and b must match."))
	end
	w = dist.weights
	
	r = zero(T)
	for i = 1 : n
		if a[i] != b[i]
			r += w[i]
		end
	end
	return r
end

hamming(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedHamming(w), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedHamming{T}, a::AbstractMatrix, b::AbstractMatrix)
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

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedHamming{T}, a::AbstractVector, b::AbstractMatrix)
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

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedHamming{T}, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedHamming{T}, a::AbstractMatrix)
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

function evaluate{T<:FloatingPoint}(dist::SqMahalanobis{T}, a::AbstractVector, b::AbstractVector)
	Q = dist.qmat
	z = a - b
	return dot(z, Q * z)
end

sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = evaluate(SqMahalanobis(Q), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::SqMahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
	Q = dist.qmat
	m, n = get_colwise_dims(size(Q, 1), r, a, b)
	z = a - b
	Qz = Q * z
	@devec r[:] = sum(Qz .* z, 1)
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::SqMahalanobis{T}, a::AbstractVector, b::AbstractMatrix)
	Q = dist.qmat
	m, n = get_colwise_dims(size(Q, 1), r, a, b)
	z = Array(T, (m, n))
	for j = 1 : n
		@devec z[:,j] = a - b[:,j]
	end
	Qz = Q * z
	@devec r[:] = sum(Qz .* z, 1)
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::SqMahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
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

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::SqMahalanobis{T}, a::AbstractMatrix)
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

function evaluate{T<:FloatingPoint}(dist::Mahalanobis{T}, a::AbstractVector, b::AbstractVector)
	sqrt(evaluate(SqMahalanobis(dist.qmat), a, b))
end

mahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix) = evaluate(Mahalanobis(Q), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::Mahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
	colwise!(r, SqMahalanobis(dist.qmat), a, b)
	@devec r[:] = sqrt(r)
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::Mahalanobis{T}, a::AbstractVector, b::AbstractMatrix)
	colwise!(r, SqMahalanobis(dist.qmat), a, b)
	@devec r[:] = sqrt(r)
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::Mahalanobis{T}, a::AbstractMatrix, b::AbstractMatrix)
	pairwise!(r, SqMahalanobis(dist.qmat), a, b)
	@devec r[:] = sqrt(max(r, 0))
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::Mahalanobis{T}, a::AbstractMatrix)
	pairwise!(r, SqMahalanobis(dist.qmat), a)
	@devec r[:] = sqrt(max(r, 0))
end


end # module end
	
	
