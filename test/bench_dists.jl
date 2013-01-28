
# This scripts compares the performance of different ways to implement column-wise 
# euclidean distances


function sqeuc_raw_forloop{T<:Real}(a::AbstractMatrix{T}, b::AbstractMatrix{T})
	# completely de-vectorized for-loop
	m, n = size(a)
	r = Array(T, n)
	for j = 1 : n
		s::T = 0
		for i = 1 : m
			v = a[i, j] - b[i, j]
			s += v * v
		end
		r[j] = sqrt(s)
	end
	return r
end

function sqeuc_sumsqr_percol{T<:Real}(a::AbstractMatrix{T}, b::AbstractMatrix{T})
	# take the some of vectorized square per column
	n = size(a, 2)
	r = Array(T, n)
	for i = 1 : n
		r[i] = sqrt(sum( (a[:,i] - b[:,i]) .^ 2 ))
	end
	return r
end

function sqeuc_norm_percol{T<:Real}(a::AbstractMatrix{T}, b::AbstractMatrix{T})
	# take norm of vectorized difference per column
	n = size(a, 2)
	r = Array(T, n)
	for i = 1 : n
		r[i] = norm(a[:,i] - b[:,i])
	end
	return r
end

function sqeuc_norm_percol_s{T<:Real}(a::AbstractMatrix{T}, b::AbstractMatrix{T})
	# take norm of vectorized difference per column (using sub)
	m, n = size(a)
	r = Array(T, n)
	for i = 1 : n
		r[i] = norm(sub(a, 1:m, i) - sub(b, 1:m, i))
	end
	return r
end

function sqeuc_map_norm{T<:Real}(a::AbstractMatrix{T}, b::AbstractMatrix{T})
	# map a norm function to each column
	r = map( i -> norm(a[:,i] - b[:,i]),  1:size(a,2) )
end


macro my_bench(f)
	quote
		# warming up
		$f(x, y)
		# timeing
		println("bench: ", $string($f))
		@time for i = 1 : 10
			$f(x, y)
		end
		println(" ")
	end
end


m = 200
n = 100000

x = rand(m, n)
y = rand(m, n)

@my_bench sqeuc_raw_forloop
@my_bench sqeuc_sumsqr_percol
@my_bench sqeuc_norm_percol
@my_bench sqeuc_norm_percol_s
@my_bench sqeuc_map_norm
