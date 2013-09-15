# Common utilities

###########################################################
#
#   helper functions for dimension checking
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


# for metrics with fixed dimension (e.g. weighted metrics)

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

