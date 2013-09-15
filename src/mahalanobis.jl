# Mahalanobis distances

type Mahalanobis{T} <: Metric
    qmat::Matrix{T}
end

type SqMahalanobis{T} <: SemiMetric
    qmat::Matrix{T}
end

result_type{T}(::Mahalanobis{T}, T1::Type, T2::Type) = T
result_type{T}(::SqMahalanobis{T}, T1::Type, T2::Type) = T

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


