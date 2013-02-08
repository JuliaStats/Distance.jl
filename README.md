# Distance.jl

A Julia package for evaluating distances(metrics) between vectors.

This package also provides carefully optimized functions to compute column-wise and pairwise distances, which is often faster than a straightforward loop implementation by one or two orders of magnitude. (See the benchmark section below for details).

**Dependencies:** [Devectorize.jl](https://github.com/lindahua/Devectorize.jl)


## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Cityblock distance 
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Correlation distance
* Chi-square distance
* Kullback-Leibler divergence
* Jensen-Shannon divergence
* Mahalanobis distance
* Squared Mahalanobis distance

For ``Euclidean distance``, ``Squared Euclidean distance``, ``Cityblock distance``, ``Minkowski distance``, and ``Hamming distance``, a weighted version is also provided.

## Basic Use

The library supports three ways of computation: *computing the distance between two vectors*, *column-wise computation*, and *pairwise computation*.


#### Computing the distance between two vectors

Each distance corresponds to a *distance type*. You can always compute a certain distance between two vectors using the following syntax

```julia
r = evaluate(dist, x, y)
```

Here, dist is an instance of a distance type. For example, the type for Euclidean distance is ``Euclidean`` (more distance types will be introduced in the next section), then you can compute the Euclidean distance between ``x`` and ``y`` as

```julia
r = evaluate(Euclidean(), x, y)
``` 

Common distances also come with convenient functions for distance evaluation. For example, you may also compute Euclidean distance between two vectors as below

```julia
r = euclidean(x, y)
```

#### Computing distances between corresponding columns

Suppose you have two ``m-by-n`` matrix ``X`` and ``Y``, then you can compute all distances between corresponding columns of X and Y in one batch, using the ``colwise`` function, as

```julia
r = colwise(dist, X, Y)
```

The output ``r`` is a vector of length ``n``. In particular, ``r[i]`` is the distance between ``X[:,i]`` and ``Y[:,i]``. The batch computation typically runs considerably faster than calling ``evaluate`` column-by-column.

Note that either of ``X`` and ``Y`` can be just a single vector -- then the ``colwise`` function will compute the distance between this vector and each column of the other parameter.

#### Computing pairwise distances

Let ``X`` and ``Y`` respectively have ``m`` and ``n`` columns. Then the ``pairwise`` function computes distances between each pair of columns in ``X`` and ``Y``:

```julia
R = pairwise(dist, X, Y)
```

In the output, ``R`` is a matrix of size ``(m, n)``, such that ``R[i,j]`` is the distance between ``X[:,i]`` and ``Y[:,j]``. Computing distances for all pairs using ``pairwise`` function is often remarkably faster than evaluting for each pair individually.

If you just want to just compute distances between columns of a matrix ``X``, you can write

```julia
R = pairwise(dist, X)
```

This statement will result in an ``m-by-m`` matrix, where ``R[i,j]`` is the distance between ``X[:,i]`` and ``X[:,j]``.
``pairwise(dist, X)`` is typically more efficient than ``pairwise(dist, X, X)``, as the former will take advantage of the symmetry when ``dist`` is a semi-metric (including metric).


#### Computing column-wise and pairwise distances inplace

If the vector/matrix to store the results are pre-allocated, you may use the storage (without creating a new array) using the following syntax:

```julia
colwise!(r, dist, X, Y)
pairwise!(R, dist, X, Y)
pairwise!(R, dist, X)
```

Please pay attention to the difference, the functions for inplace computation are ``colwise!`` and ``pairwise!`` (instead of ``colwise`` and ``pairwise``).



## Distance type hierarchy

The distances are organized into a type hierarchy. 

At the top of this hierarchy is an abstract class **PreMetric**, which is defined to be a function ``d`` that satisfies

	d(x, x) == 0  for all x
	d(x, y) >= 0  for all x, y
	
**SemiMetric** is a abstract type that refines **PreMetric**. Formally, a *semi-metric* is a *pre-metric* that is also symmetric, as

	d(x, y) == d(y, x)  for all x, y
	
**Metric** is a abstract type that further refines **SemiMetric**. Formally, a *metric* is a *semi-metric* that also satisfies triangle inequality, as

	d(x, z) <= d(x, y) + d(y, z)  for all x, y, z
	
This type system has practical significance. For example, when computing pairwise distances between a set of vectors, you may only perform computation for half of the pairs, and derive the values immediately for the remaining halve by leveraging the symmetry of *semi-metrics*.

Each distance corresponds to a distance type. The type name and the corresponding mathematical definitions of the distances are listed in the following table.

| type name            |  convenient syntax   | math definition     | 
| -------------------- | -------------------- | --------------------|
|  Euclidean           |  euclidean(x, y)     | sqrt(sum((x - y) .^ 2)) |
|  SqEuclidean         |  sqeuclidean(x, y)   | sum((x - y).^2) |
|  Cityblock           |  cityblock(x, y)     | sum(abs(x - y)) |
|  Chebyshev           |  chebyshev(x, y)     | max(abs(x - y)) |
|  Minkowski           |  minkowski(x, y, p)  | sum(abs(x - y).^p) ^ (1/p) |
|  Hamming             |  hamming(x, y)       | sum(x .!= y) |
|  CosineDist          |  cosine_dist(x, y)   | 1 - dot(x, y) / (norm(x) * norm(y)) |
|  CorrDist            |  corr_dist(x, y)     | cosine_dist(x - mean(x), y - mean(y)) |
|  ChiSqDist           |  chisq_dist(x, y)    | sum((x - y).^2 / (x + y)) | 
|  KLDivergence        |  kl_divergence(x, y) | sum(p .* log(p ./ q)) |
|  JSDivergence        |  js_divergence(x, y) | KL(x, m) / 2 + KL(y, m) / 2 with m = (x + y) / 2 |
|  Mahalanobis         |  mahalanobis(x, y, Q)    | sqrt((x - y)' * Q * (x - y)) |
|  SqMahalanobis       |  sqmahalanobis(x, y, Q)  |  (x - y)' * Q * (x - y)  |
|  WeightedEuclidean   |  euclidean(x, y, w)      | sqrt(sum((x - y).^2 .* w))  |
|  WeightedSqEuclidean |  sqeuclidean(x, y, w)    | sum((x - y).^2 .* w)  |
|  WeightedCityblock   |  cityblock(x, y, w)      | sum(abs(x - y) .* w)  |
|  WeightedMinkowski   |  minkowski(x, y, w, p)   | sum(abs(x - y).^p .* w) ^ (1/p)  |
|  WeightedHamming     |  hamming(x, y, w)        | sum((x .!= y) .* w)  |
  
**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way.


## Benchmarks


The implementation has been carefully optimized based on benchmarks. The Julia scripts ``test/bench_colwise.jl`` and ``test/bench_pairwise.jl`` run the benchmarks on a variety of distances, respectively under column-wise and pairwise settings.

Here are the benchmarks that I obtained on Mac OS X 10.8 with Intel Core i7 2.6 GHz.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distance.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance   |   loop  |   colwise   |   gain     |
|------------ | --------| ------------| -----------|
| SqEuclidean | 0.038312 | 0.004708 | 8.137x | 
| Euclidean | 0.036947 | 0.004158 | 8.885x | 
| Cityblock | 0.037507 | 0.004348 | 8.626x |
| Chebyshev | 0.045246 | 0.012861 | 3.517x |
| Minkowski | 0.418969 | 0.379957 | 1.103x |
| Hamming | 0.035414 | 0.004264 | 8.305x |
| CosineDist | 0.053191 | 0.008009 | 6.642x |
| CorrDist | 0.085048 | 0.035571 | 2.391x |
| ChiSqDist | 0.04407 | 0.00839 | 5.253x |
| KLDivergence | 0.071618 | 0.040244 | 1.780x |
| JSDivergence | 0.45729 | 0.417977 | 1.094x |
| WeightedSqEuclidean | 0.040023 | 0.006049 | 6.617x |
| WeightedEuclidean | 0.039938 | 0.005953 | 6.710x |
| WeightedCityblock | 0.038502 | 0.006362 | 6.052x |
| WeightedMinkowski | 0.540418 | 0.510852 | 1.058x |
| WeightedHamming | 0.039012 | 0.004541 | 8.592x |
| SqMahalanobis | 0.135427 | 0.040468 | 3.347x |
| Mahalanobis | 0.135613 | 0.04203 | 3.227x |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 9x), especially when the internal computation of each distance is simple. Nonetheless, when the computaton of a single distance is heavy enough (e.g. *Minkowski* and *JSDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distance.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance   |   loop  |   colwise   |   gain     |
|------------ | --------| ------------| -----------|
| SqEuclidean | 0.092455 | 0.000459 | **201.31x** |
| Euclidean | 0.091295 | 0.000734 | **124.36x** |
| Cityblock | 0.09096 | 0.012827 | 7.0913x |
| Chebyshev | 0.105589 | 0.033345 | 3.1665x | 
| Minkowski | 1.015888 | 0.940429 | 1.0802x |
| Hamming | 0.086808 | 0.010143 | 8.5583x |
| CosineDist | 0.130899 | 0.001278 | **102.41x** |
| CorrDist | 0.212448 | 0.000889 | **239.11x** |
| ChiSqDist | 0.110598 | 0.025816 | 4.2841x |
| KLDivergence | 0.177114 | 0.103134 | 1.7173x |
| JSDivergence | 1.135585 | 1.030385 | 1.1021x |
| WeightedSqEuclidean | 0.101272 | 0.00064 | **158.18x** |
| WeightedEuclidean | 0.103933 | 0.000994 | **104.53x** |
| WeightedCityblock | 0.101804 | 0.017774 | 5.7278x |
| WeightedMinkowski | 1.381289 | 1.30307 | 1.0600x |
| WeightedHamming | 0.095391 | 0.011416 | 8.3559x |
| SqMahalanobis | 0.348651 | 0.000758 | **460.14x** |
| Mahalanobis | 0.360126 | 0.001236 | **291.48x** |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).


