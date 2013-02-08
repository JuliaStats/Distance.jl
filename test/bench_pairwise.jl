
# Benchmark on pairwise distance evaluation

using Distance

macro bench_pairwise_dist(repeat, dist, x, y)
	quote
		println("bench ", typeof($dist))
	
		# warming up
		r1 = evaluate($dist, ($x)[:,1], ($y)[:,1])
		pairwise($dist, $x, $y)
		
		# timing
		
		t0 = @elapsed for k = 1 : $repeat
			m = size($x, 1)
			n = size($x, 2)
			r = Array(typeof(r1), (m, n))
			for j = 1 : n
				for i = 1 : m
					r[i, j] = evaluate($dist, ($x)[:,i], ($y)[:,j])
				end
			end
		end
		@printf "    loop:      t = %9.6fs\n" (t0 / $repeat) 
		
		t1 = @elapsed for k = 1 : $repeat
			r = pairwise($dist, $x, $y)
		end
		@printf "    pairwise:  t = %9.6fs  |  gain = %7.4fx\n" (t1 / $repeat) (t0 / t1)
		println()
	end
end


m = 200
n = 500

x = rand(m, n)
y = rand(m, n)

@bench_pairwise_dist 20 SqEuclidean() x y
@bench_pairwise_dist 20 Euclidean() x y
@bench_pairwise_dist 20 Cityblock() x y
@bench_pairwise_dist 20 Chebyshev() x y
@bench_pairwise_dist 2 Minkowski(3.0) x y
@bench_pairwise_dist 10 CosineDist() x y
@bench_pairwise_dist 10 CorrDist() x y
@bench_pairwise_dist 5 KLDivergence() x y
@bench_pairwise_dist 2 JSDivergence() x y
