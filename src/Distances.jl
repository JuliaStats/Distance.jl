module Distances
	
export 
	# generic types/functions
	GeneralizedDistance, 
	Distance,
	cdist,
	pdist,
	
	# distance classes
	EuclideanDistance,
	SquaredEuclideanDistance,
	CityblockDistance,
	ChebyshevDistance,
	MinkowskiDistance,
	HammingDistance,
	CosineDistance,
	KullbackLeiblerDivergence,
	JensenShannonDivergence,
	
	# convenient functions
	euclidean_dist,
	sqeuclidean_dist,
	cityblock_dist,
	chebyshev_dist,
	minkowski_dist,
	hamming_dist,
	cosine_dist,
	kl_div,
	js_div
	
	
##########################################################################
#
#  The type hierarchy for distances
#
##########################################################################

# any thing that people typically think of as a kind of distance 
# that takes two vectors and outputs a real scalar
# e.g. K-L divergence
abstract GeneralizedDistance

# distance that satisfies d(x, x) == 0 and d(x, y) == d(y, x)
# pairwise computation can take advantage of these properties to save computation
abstract Distance <: GeneralizedDistance

# specific distance types

type EuclideanDistance <: Distance 
end

type SquaredEuclideanDistance <: Distance 
end

type CityblockDistance <: Distance
end

type ChebyshevDistance <: Distance
end

type MinkowskiDistance <: Distance
	p::Number
end

type HammingDistance <: Distance
end

type CosineDistance <: Distance
end

type KullbackLeiblerDivergence <: GeneralizedDistance
end

type JensenShannonDivergence <: Distance
end



##########################################################################
#
#  General cdist implementation
#
#  Remarks
#  --------
#  cdist(dist, a::Vector, b::Vector) should be implemented respectively
#  for each distance.	
#
#  This general implementation extends them to matrices. 
#
##########################################################################

function cdist(dist::GeneralizedDistance, a::AbstractVector, b::AbstractMatrix)
	m, n = size(b)
	
	# calculate the first distance (to determine value type)
	
	d1 = cdist(dist, a, b[:,1])
	r = Array(typeof(d1), n)
	r[1] = d1
	
	# calculate the rest
	
	for i = 2 : n
		r[i] = cdist(dist, a, b[:,i])
	end
	
	return r
end

function cdist(dist::GeneralizedDistance, a::AbstractMatrix, b::AbstractVector)
	m, n = size(a)
	
	# calculate the first distance (to determine value type)
	
	d1 = cdist(dist, a[:,1], b)
	r = Array(typeof(d1), n)
	r[1] = d1
	
	# calculate the rest
	
	for i = 2 : n
		r[i] = cdist(dist, a[:,i], b)
	end
	
	return r
end

function cdist(dist::GeneralizedDistance, a::AbstractMatrix, b::AbstractMatrix)
	@assert size(a, 2) == size(b, 2)
	n = size(a, 2)
	
	# calculate the first distance (to determine distance value type)
	
	d1 = cdist(dist, a[:,1], b[:,1])
	r = Array(typeof(d1), n)
	r[1] = d1
	
	# calculate the rest
	
	for i = 2 : n
		r[i] = cdist(dist, a[:,i], b[:,i])
	end
	
	return r
end


##########################################################################
#
#  General pdist implementation
#
##########################################################################

function pdist(dist::GeneralizedDistance, a::Matrix, b::Matrix)
	m = size(a, 2)
	n = size(b, 1)
	
	# calculate the first distance (to determine distance value type)
	
	d1 = cdist(dist, a[:,1], b[:,1])
	r = Array(typeof(d1), (m, n))
	r[1, 1] = d1
	
	# calculate the rest
	
	for i = 2 : m
		b1 = b[:,1]
		r[i, 1] = cdist(dist, a[:,i], b1)
	end
	
	for j = 2 : n
		bj = b[:,j]
		for i = 1 : m
			r[i, j] = cdist(dist, a[:,i], bj)
		end
	end
	
	return r
end

pdist(dist::GeneralizedDistance, a::Matrix) = pdist(dist, a, a)

function pdist(dist::Distance, a::Matrix)
	n = size(a, 2)
	
	# calculate the first distance (to determine distance value type)
	
	d1 = cdist(dist, a[:,1], b[:,1])
	r = Array(typeof(d1), (n, n))
	r[1, 1] = 0
	
	# calculate the rest
	
	for i = 2 : n
		b1 = b[:,1]
		r[i, 1] = cdist(dist, a[:,i], b1)
	end
	
	for j = 2 : n
		bj = b[:,j]
		
		for i = 1 : j-1
			r[i, j] = r[j, i]
		end
		
		r[j, j] = 0
				
		for i = j+1 : n	
			r[i, j] = cdist(dist, a[:,i], bj)
		end
	end
	
	return r
end




##########################################################################
#
#  Implementation of distance functions
#
#  Most implementations use de-vectorized codes for the sake of
#  run-time performance.
#
##########################################################################

# euclidean 

function cdist{T<:Real}(dist::EuclideanDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n
		v = a[i] - b[i]
		s += v * v
	end
	return sqrt(s)
end

# squared euclidean

function cdist{T<:Real}(dist::SquaredEuclideanDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n 
		v = a[i] - b[i]
		s += v * v
	end
	return s
end	

# cityblock
	
function cdist{T<:Real}(dist::CityblockDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n  
		s += abs(a[i] - b[i])
	end
	return s
end	

# chebyshev
	
function cdist{T<:Real}(dist::ChebyshevDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n 
		s = max(s, abs(a[i] - b[i]))		
	end
	return s
end
	
# minkowski
	
function cdist{T<:Real}(dist::MinkowskiDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n 
		s += abs(a[i] - b[i]) ^ dist.p
	end
	return s ^ (1 / dist.p)
end	
		
# hamming

function cdist{T<:Real}(dist::HammingDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::Int = 0
	n = length(a)
	for i = 1 : n  
		if (a[i] != b[i]) 
			s += 1   
		end
	end
	return s
end

# cosine

function cdist{T<:Real}(dist::CosineDistance, a::AbstractVector{T}, b::AbstractVector{T})
	xx::T = 0
	xy::T = 0
	yy::T = 0
	n = length(a)
	for i = 1 : n  
		ai = a[i]
		bi = b[i]
		xx += ai * ai
		xy += ai * bi
		yy += bi * bi
	end
	return xy / (sqrt(xx) * sqrt(yy))
end

# K-L divergence

xlogx(x::Real) = x > 0 ? x * log(x) : zero(x)

function cdist{T<:Real}(dist::KullbackLeiblerDivergence, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n  
		ai = a[i]
		if ai > 0
			s += ai * (log(ai) - log(b[i]))
		end		
	end
	return s
end

# J-S divergence

function cdist{T<:Real}(dist::JensenShannonDivergence, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n  
		ai = a[i]
		bi = b[i]
		vi = (ai + bi) / 2
		s += ( (xlogx(ai) + xlogx(bi)) / 2 - xlogx(vi) )
	end
	return s
end


##########################################################################
#
#  Convenient functions
#
##########################################################################
	
euclidean_dist(a::AbstractArray, b::AbstractArray)   = cdist(EuclideanDistance(), a, b)	
sqeuclidean_dist(a::AbstractArray, b::AbstractArray) = cdist(SquaredEuclideanDistance(), a, b)
cityblock_dist(a::AbstractArray, b::AbstractArray)   = cdist(CityblockDistance(), a, b)	
chebyshev_dist(a::AbstractArray, b::AbstractArray)   = cdist(ChebyshevDistance(), a, b)	

minkowski_dist(a::AbstractArray, b::AbstractArray, p::Number)   = cdist(MinkowskiDistance(p), a, b)	

hamming_dist(a::AbstractArray, b::AbstractArray) = cdist(HammingDistance(), a, b)	
cosine_dist(a::AbstractArray, b::AbstractArray)  = cdist(CosineDistance(), a, b)

kl_div(a::AbstractArray, b::AbstractArray) = cdist(KullbackLeiblerDivergence(), a, b)
js_div(a::AbstractArray, b::AbstractArray) = cdist(JensenShannonDivergence(), a, b)

end # module end	
	
	