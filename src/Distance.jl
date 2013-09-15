module Distance

using Devectorize

export
    # generic types/functions
    GeneralizedMetric,
    Metric,

    # distance classes
    Euclidean,
    SqEuclidean,
    Cityblock,
    Chebyshev,
    Minkowski,

    Hamming,
    CosineDist,
    CorrDist,
    ChiSqDist,
    KLDivergence,
    JSDivergence,

    WeightedEuclidean,
    WeightedSqEuclidean,
    WeightedCityblock,
    WeightedMinkowski,
    WeightedHamming,
    SqMahalanobis,
    Mahalanobis,

    # convenient functions
    euclidean,
    sqeuclidean,
    cityblock,
    chebyshev,
    minkowski,
    mahalanobis,

    hamming,
    cosine_dist,
    corr_dist,
    chisq_dist,
    kl_divergence,
    js_divergence,

    weighted_euclidean,
    weighted_sqeuclidean,
    weighted_cityblock,
    weighted_minkowski,
    weighted_hamming,
    sqmahalanobis,
    mahalanobis,

    # generic functions
    result_type,
    colwise,
    pairwise,
    colwise!,
    pairwise!,
    evaluate,

    # other convenient functions
    At_Q_B, At_Q_A


include("common.jl")
include("generic.jl")
include("at_q_b.jl")
include("metrics.jl")
include("wmetrics.jl")
include("mahalanobis.jl")

end # module end


