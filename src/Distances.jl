module Distances
	
export 
	GeneralizedDistance, 
	Distance,
	cdist,
	pdist,
	EuclideanDistance,
	SquaredEuclideanDistance,
	eucdist,
	sqeucdist
	
##########################################################################
#
#  The type hierarchy for distances
#
##########################################################################

abstract GeneralizedDistance
abstract Distance <: GeneralizedDistance

type EuclideanDistance <: Distance 
end

type SquaredEuclideanDistance <: Distance 
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
		r[i] = cdist(dist, sub(a, 1:m, i), b)
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
		r[i, j] = cdist(dist, a[:,i], bj)
	end
	
	return r
end

##########################################################################
#
#  Specific distances
#
##########################################################################

function cdist{T<:Real}(dist::EuclideanDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n  # devectorize to make it faster
		v = a[i] - b[i]
		s += v * v
	end
	return sqrt(s)
end

function cdist{T<:Real}(dist::SquaredEuclideanDistance, a::AbstractVector{T}, b::AbstractVector{T})
	s::T = 0
	n = length(a)
	for i = 1 : n  # devectorize to make it faster
		v = a[i] - b[i]
		s += v * v
	end
	return s
end	
	

##########################################################################
#
#  Convenient functions
#
##########################################################################
	
eucdist(a::AbstractArray, b::AbstractArray)   = cdist(EuclideanDistance(), a, b)	
sqeucdist(a::AbstractArray, b::AbstractArray) = cdist(SquaredEuclideanDistance(), a, b)
		
	
end # module end	
	
	