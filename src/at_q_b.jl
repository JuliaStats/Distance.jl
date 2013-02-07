
using Devectorize

###########################################################
#
#   Computation of quadratic form
#
# 	if Q is a vector
#
#	 	At_Q_B(Q, a, b) = a' * diagm(Q) * b
# 		At_Q_A(Q, a)    = a' * diagm(Q) * a
#
# 	if Q is a matrix
#
#		At_Q_B(Q, a, b) = a' * Q * b
#		At_Q_B(Q, a)    = a' * Q * b
#
#	inplace version is also provided
#
#		At_Q_B!(r, Q, a, b)
#		At_Q_A!(r, Q, a)
#
#	When both a and b are vectors, it returns a scalar;
#	When a or b is vector, the other is matrix, it returns
#	a vector;
#	When both a and b are matrices, it returns a matrix
# 
###########################################################


# dimension checking helpers (for At_Q_B)

function atqb_check_dims(Q::Vector, a::VecOrMat, b::VecOrMat)
	if !(length(Q) == size(a,1) == size(b,1))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function atqb_check_dims(Q::Vector, a::VecOrMat)
	if !(length(Q) == size(a,1))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function atqb_check_dims(Q::Matrix, a::VecOrMat, b::VecOrMat)
	if !(size(Q, 1) == size(a,1) && size(Q, 2) == size(b,1))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function atqb_check_dims(Q::Matrix, a::VecOrMat)
	if !(size(Q, 1) == size(Q, 2) == size(a,1))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function atqb_check_rsize(r::Array, len::Int)
	if length(r) != len
		throw(ArgumentError("Incorrect length of r."))
	end
end

function atqb_check_rsize(r::Array, siz::(Int, Int))
	if !(size(r) == siz)
		throw(ArgumentError("Incorrect size of r."))
	end
end


# At_Q_B & At_Q_A (Q is a vector)

function At_Q_B{T<:FloatingPoint}(Q::Vector{T}, a::Vector{T}, b::Vector{T})
	@devec r = sum(Q .* a .* b)
	return r
end

function At_Q_A{T<:FloatingPoint}(Q::Vector{T}, a::Vector{T})
	@devec r = sum(Q .* sqr(a))
	return r
end

function At_Q_B!{T<:FloatingPoint}(r::Array{T}, Q::Vector{T}, a::Vector{T}, b::Matrix{T})
	atqb_check_dims(Q, a, b)
	n = size(b, 2)
	atqb_check_rsize(r, n)
	
	for j = 1 : n
		@devec r[j] = sum(a .* Q .* b[:,j])
	end
end

function At_Q_B!{T<:FloatingPoint}(r::Array{T}, Q::Vector{T}, a::Matrix{T}, b::Vector{T})
	atqb_check_dims(Q, a, b)
	n = size(a, 2)
	atqb_check_rsize(r, n)
	
	for j = 1 : n
		@devec r[j] = sum(a[:,j] .* Q .* b)
	end
end

function At_Q_B!{T<:FloatingPoint}(r::Matrix{T}, Q::Vector{T}, a::Matrix{T}, b::Matrix{T})
	atqb_check_dims(Q, a, b)
	m = size(a, 2)
	n = size(b, 2)
	atqb_check_rsize(r, (m, n))
	
	t = similar(b)
	for j = 1 : n
		@devec t[:,j] = b[:,j] .* Q
	end
	At_mul_B(r, a, t)
end

function At_Q_A!{T<:FloatingPoint}(r::Matrix{T}, Q::Vector{T}, a::Matrix{T})
	atqb_check_dims(Q, a)
	n = size(a, 2)
	atqb_check_rsize(r, (n, n))
	
	t = similar(a)
	for j = 1 : n
		@devec t[:,j] = a[:,j] .* Q
	end
	At_mul_B(r, a, t)
end


function At_Q_B{T<:FloatingPoint}(Q::Vector{T}, a::Vector{T}, b::Matrix{T})
	r = Array(T, size(b, 2))
	At_Q_B!(r, Q, a, b)
	return r
end

function At_Q_B{T<:FloatingPoint}(Q::Vector{T}, a::Matrix{T}, b::Vector{T})
	r = Array(T, size(a, 2))
	At_Q_B!(r, Q, a, b)
	return r
end

function At_Q_B{T<:FloatingPoint}(Q::Vector{T}, a::Matrix{T}, b::Matrix{T})
	r = Array(T, (size(a, 2), size(b, 2)))
	At_Q_B!(r, Q, a, b)
	return r
end

function At_Q_A{T<:FloatingPoint}(Q::Vector{T}, a::Matrix{T})
	n = size(a, 2)
	r = Array(T, (n, n))
	At_Q_A!(r, Q, a)
	return r
end


# At_Q_B & At_Q_A (Q is a matrix)


function At_Q_B{T<:FloatingPoint}(Q::Matrix{T}, a::Vector{T}, b::Vector{T})
	dot(a, Q * b)
end

function At_Q_A{T<:FloatingPoint}(Q::Matrix{T}, a::Vector{T})
	dot(a, Q * a)
end

function At_Q_B!{T<:FloatingPoint}(r::Array{T}, Q::Matrix{T}, a::Vector{T}, b::Matrix{T})
	atqb_check_dims(Q, a, b)
	n = size(b, 2)
	atqb_check_rsize(r, n)
	
	qta = Array(T, size(Q, 2))
	At_mul_B(qta, Q, a)
	for j = 1 : n
		@devec r[j] = sum(qta .* b[:,j])
	end
end

function At_Q_B!{T<:FloatingPoint}(r::Array{T}, Q::Matrix{T}, a::Matrix{T}, b::Vector{T})
	atqb_check_dims(Q, a, b)
	n = size(a, 2)
	atqb_check_rsize(r, n)
	
	qb = Q * b
	for j = 1 : n
		@devec r[j] = sum(a[:,j] .* qb)
	end
end

function At_Q_B!{T<:FloatingPoint}(r::Matrix{T}, Q::Matrix{T}, a::Matrix{T}, b::Matrix{T})
	atqb_check_dims(Q, a, b)
	m = size(a, 2)
	n = size(b, 2)
	atqb_check_rsize(r, (m, n))
	
	At_mul_B(r, a, Q * b)
end

function At_Q_A!{T<:FloatingPoint}(r::Matrix{T}, Q::Matrix{T}, a::Matrix{T})
	atqb_check_dims(Q, a)
	n = size(a, 2)
	atqb_check_rsize(r, (n, n))
	
	At_mul_B(r, a, Q * a)
end

function At_Q_B{T<:FloatingPoint}(Q::Matrix{T}, a::Vector{T}, b::Matrix{T})
	r = Array(T, size(b, 2))
	At_Q_B!(r, Q, a, b)
	return r
end

function At_Q_B{T<:FloatingPoint}(Q::Matrix{T}, a::Matrix{T}, b::Vector{T})
	r = Array(T, size(a, 2))
	At_Q_B!(r, Q, a, b)
	return r
end

function At_Q_B{T<:FloatingPoint}(Q::Matrix{T}, a::Matrix{T}, b::Matrix{T})
	r = Array(T, (size(a, 2), size(b, 2)))
	At_Q_B!(r, Q, a, b)
	return r
end

function At_Q_A{T<:FloatingPoint}(Q::Matrix{T}, a::Matrix{T})
	n = size(a, 2)
	r = Array(T, (n, n))
	At_Q_A!(r, Q, a)
	return r
end


# dimension checking helpers (for A_Q_Bt)

function aqbt_check_dims(Q::Vector, a::VecOrMat, b::VecOrMat)
	if !(length(Q) == size(a,2) == size(b,2))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function aqbt_check_dims(Q::Vector, a::VecOrMat)
	if !(length(Q) == size(a,2))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function aqbt_check_dims(Q::Matrix, a::VecOrMat, b::VecOrMat)
	if !(size(Q,1) == size(a,2) && size(Q,2) == size(b,2))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function aqbt_check_dims(Q::Matrix, a::VecOrMat)
	if !(size(Q,1) == size(Q,2) == size(a,2))
		throw(ArgumentError("Mismatched dimensions."))
	end
end

function aqbt_check_rsize(r::Array, len::Int)
	if length(r) != len
		throw(ArgumentError("Incorrect length of r."))
	end
end

function aqbt_check_rsize(r::Array, siz::(Int, Int))
	if !(size(r) == siz)
		throw(ArgumentError("Incorrect size of r."))
	end
end


