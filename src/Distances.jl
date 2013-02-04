module Metrics
	
export 
	# generic types/functions
	GeneralizedMetric, 
	Metric,
	cdist,
	pdist,
	
	# distance classes
	Euclidean,
	SqEuclidean,
	Chebyshev,
	Cityblock,
	Minkowski,
	Hamming,
	CosineDist,
	CorrDist,
	KLDivergence,
	JSDivergence,
	
	# convenient functions
	euclidean,
	sqeuclidean,
	chebyshev,
	cityblock,
	minkowski,
	hamming,
	cosine_dist,
	corr_dist,
	kl_divergence,
	js_divergence,

	# generic functions 
	cdist,
	pdist


###########################################################
#
#	Metric types
#
###########################################################

abstract PreMetric
abstract SemiMetric <: PreMetric
abstract Metric <: SemiMetric

type Euclidean <: Metric end
type SqEuclidean <: SemiMetric end
type Chebyshev <: Metric end
type Cityblock <: Metric end
type Minkowski <: Metric end
type Hamming <: Metric end
type CosineDist <: SemiMetric end
type CorrDist <: SemiMetric end
type KLDivergence <: PreMetric end
type JSDivergence <: SemiMetric end

end # module end	
	
	