tests = ["atqb", "dists", "custom_dists"]

for t in tests
	fn = joinpath("test", "test_$t.jl")
	println("$fn ...")
	include(fn)
end
