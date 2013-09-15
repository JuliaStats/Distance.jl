# Weighted metrics


###########################################################
#
#   Metric types
#
###########################################################

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

result_type{T}(::WeightedEuclidean{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedSqEuclidean{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedCityblock{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedMinkowski{T}, T1::Type, T2::Type) = T
result_type{T}(::WeightedHamming{T}, T1::Type, T2::Type) = T


###########################################################
#
#   Specialized distances
#
###########################################################


# Weighted squared Euclidean

evaluate{T<:FloatingPoint}(dist::WeightedSqEuclidean{T}, a::AbstractVector, b::AbstractVector) = wsqdiffsum(dist.weights, a, b)
wsqeuclidean(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedSqEuclidean(w), a, b)

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedSqEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
    w = dist.weights
    m, na, nb = get_pairwise_dims(length(w), r, a, b)

    sa2 = wsqsum(w, a, 1)
    sb2 = wsqsum(w, b, 1)
    At_Q_B!(r, w, a, b)
    for j = 1 : nb
        for i = 1 : na
            @inbounds r[i,j] = sa2[i] + sb2[j] - 2 * r[i,j]
        end
    end
    r
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedSqEuclidean{T}, a::AbstractMatrix)
    w = dist.weights
    m, n = get_pairwise_dims(length(w), r, a)

    sa2 = wsqsum(w, a, 1)
    At_Q_A!(r, w, a)
    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
        end
        @inbounds r[j,j] = 0
        for i = j+1 : n
            @inbounds r[i,j] = sa2[i] + sa2[j] - 2 * r[i,j]
        end
    end
    r
end


# Weighted Euclidean

function evaluate{T<:FloatingPoint}(dist::WeightedEuclidean{T}, a::AbstractVector, b::AbstractVector)
    sqrt(evaluate(WeightedSqEuclidean(dist.weights), a, b))
end

weuclidean(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedEuclidean(w), a, b)

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function colwise!{T<:FloatingPoint}(r::AbstractArray, dist::WeightedEuclidean{T}, a::AbstractVector, b::AbstractMatrix)
    sqrt!(colwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedEuclidean{T}, a::AbstractMatrix, b::AbstractMatrix)
    sqrt!(pairwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function pairwise!{T<:FloatingPoint}(r::AbstractMatrix, dist::WeightedEuclidean{T}, a::AbstractMatrix)
    sqrt!(pairwise!(r, WeightedSqEuclidean(dist.weights), a))
end


# Weighted Cityblock

evaluate{T<:FloatingPoint}(dist::WeightedCityblock{T}, a::AbstractVector, b::AbstractVector) = wadiffsum(dist.weights, a, b)
wcityblock(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedCityblock(w), a, b)


# WeightedMinkowski

function evaluate{T<:FloatingPoint}(dist::WeightedMinkowski{T}, a::AbstractVector, b::AbstractVector) 
    wsum_fdiff(dist.weights, FixAbsPow(dist.p), a, b) ^ inv(dist.p)
end

wminkowski(a::AbstractVector, b::AbstractVector, w::AbstractVector, p::Real) = evaluate(WeightedMinkowski(w, p), a, b)


# WeightedHamming

function evaluate{T<:FloatingPoint}(dist::WeightedHamming{T}, a::AbstractVector, b::AbstractVector)
    n = length(a)
    w = dist.weights

    r = zero(T)
    for i = 1 : n
        @inbounds if a[i] != b[i]
            r += w[i]
        end
    end
    return r
end

whamming(a::AbstractVector, b::AbstractVector, w::AbstractVector) = evaluate(WeightedHamming(w), a, b)
