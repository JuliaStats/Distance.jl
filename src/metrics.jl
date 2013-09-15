# Ordinary metrics

###########################################################
#
#   Metric types
#
###########################################################

type Euclidean <: Metric end
type SqEuclidean <: SemiMetric end
type Chebyshev <: Metric end
type Cityblock <: Metric end

type Minkowski <: Metric
    p::Real
end

type Hamming <: Metric end
result_type(::Hamming, T1::Type, T2::Type) = Int

type CosineDist <: SemiMetric end
type CorrDist <: SemiMetric end

type ChiSqDist <: SemiMetric end
type KLDivergence <: PreMetric end
type JSDivergence <: SemiMetric end


###########################################################
#
#   Specialized distances
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
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
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
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
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

