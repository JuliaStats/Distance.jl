
# Benchmark on column-wise distance evaluation

using Distance

function sqeuc_pw(a::Matrix, b::Matrix)
	r = pairwise(SqEuclidean(), a, b)
end

function euc_pw(a::Matrix, b::Matrix)
	r = pairwise(Euclidean(), a, b)
end


macro bench_colwise_dist(dist, x, y)
	quote
		println("bench ", typeof($dist))
	
		# warming up
		r1 = evaluate($dist, ($x)[:,1], ($y)[:,1])
		colwise($dist, $x, $y)
		
		# timing
		
		repeat = 20
		
		t0 = @elapsed for k = 1 : repeat
			n = size($x, 2)
			r = Array(typeof(r1), n)
			for j = 1 : n
				($x)[j] = evaluate($dist, ($x)[:,j], ($y)[:,j])
			end
		end
		@printf "    loop:     t = %9.6fs\n" (t0 / repeat) 
		
		t1 = @elapsed for k = 1 : repeat
			r = colwise($dist, $x, $y)
		end
		@printf "    colwise:  t = %9.6fs  |  gain = %7.4fx\n" (t1 / repeat) (t0 / t1)
		println()
	end
end


m = 200
n = 10000

x = rand(m, n)
y = rand(m, n)

@bench_colwise_dist SqEuclidean() x y
@bench_colwise_dist Euclidean() x y
@bench_colwise_dist Cityblock() x y
@bench_colwise_dist Chebyshev() x y
@bench_colwise_dist Minkowski(3.0) x y
@bench_colwise_dist Hamming() x y
@bench_colwise_dist CosineDist() x y
@bench_colwise_dist CorrDist() x y
@bench_colwise_dist KLDivergence() x y
@bench_colwise_dist JSDivergence() x y
