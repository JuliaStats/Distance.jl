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
