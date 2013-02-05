
# This scripts compares the performance of different ways to implement column-wise 
# euclidean distances

using Metrics

function sqeuc_pw(a::Matrix, b::Matrix)
	r = pairwise(SqEuclidean(), a, b)
end

function euc_pw(a::Matrix, b::Matrix)
	r = pairwise(Euclidean(), a, b)
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


m = 1000
n = 1000

x = rand(m, n)
y = rand(m, n)

@my_bench sqeuc_pw
#@my_bench euc_pw
