# Bhattacharyya distances. Much like for KLDivergence we assume the vectors to
# be compared are probability distributions, frequencies or counts rather than
# vectors of samples. Pre-calc accordingly.

type BhattacharyyaCoeff <: SemiMetric end

type BhattacharyyaDist <: SemiMetric end

type HellingerDist <: Metric end


# Bhattacharyya coefficient

function evaluate{T<:Number}(dist::BhattacharyyaCoeff, a::AbstractVector{T}, b::AbstractVector{T})
    n = length(a)
    sqab = zero(T)
    # We must normalize since we cannot assume that the vectors are normalized to probability vectors.
    asum = zero(T)
    bsum = zero(T)

    for i = 1:n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        sqab += sqrt(ai * bi)
        asum += ai
        bsum += bi
    end

    sqab / sqrt(asum * bsum) 
end

BC = BhattacharyyaCoeff() # Create one so we need not create new ones on each invocation

bhattacharyya_coeff(a::AbstractVector, b::AbstractVector) = evaluate(BC, a, b)
evaluate{T <: Number}(dist::BhattacharyyaCoeff, a::T, b::T) = throw("Bhattacharyya coefficient cannot be calculated for scalars")
bhattacharyya_coeff{T <: Number}(a::T, b::T) = evaluate(BC, a, b)

# Faster pair- and column-wise versions TBD...


# Bhattacharyya distance
evaluate{T<:Number}(dist::BhattacharyyaDist, a::AbstractVector{T}, b::AbstractVector{T}) = -log(evaluate(BC, a, b))
bhattacharyya(a::AbstractVector, b::AbstractVector) = evaluate(BhattacharyyaDist(), a, b)
evaluate{T <: Number}(dist::BhattacharyyaDist, a::T, b::T) = throw("Bhattacharyya distance cannot be calculated for scalars")
bhattacharyya{T <: Number}(a::T, b::T) = evaluate(BhattacharyyaDist(), a, b)

# Hellinger distance
evaluate{T<:Number}(dist::HellingerDist, a::AbstractVector{T}, b::AbstractVector{T}) = sqrt(1 - evaluate(BC, a, b))
hellinger(a::AbstractVector, b::AbstractVector) = evaluate(HellingerDist(), a, b)
evaluate{T <: Number}(dist::BhattacharyyaDist, a::T, b::T) = throw("Hellinger distance cannot be calculated for scalars")
hellinger{T <: Number}(a::T, b::T) = evaluate(HellingerDist(), a, b)
