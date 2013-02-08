# Metrics

Evaluation of distances(metrics) between vectors.

## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Cityblock distance 
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Correlation distance
* Kullback-Leibler divergence
* Jensen-Shannon divergence
* Mahalanobis distance

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

| type name            |  math definition     | 
| -------------------- | -------------------- |
|  Euclidean           |  sqrt(sum((x - y) .^ 2)) |
|  SqEuclidean         |  sum((x - y).^2) |
|  Cityblock           |  sum(abs(x - y)) |
|  Chebyshev           |  max(abs(x - y)) |
|  Minkowski           |  sum(abs(x - y).^p) ^ (1/p) |
|  Hamming             |  sum(x .!= y) |
|  CosineDist          |  1 - dot(x, y) / (norm(x) * norm(y)) |
|  CorrDist            |  1 - dot(u, v) / (norm(u) * norm(v)), u = x - mean(x), v = y - mean(y) |
|  KLDivergence        |  sum(p .* log(p ./ q)) |
|  JSDivergence        |  KL(x, m) / 2 + KL(y, m) / 2 with m = (x + y) / 2 |
|  Mahalanobis         |  sqrt((x - y)' * Q * (x - y)) |
|  SqMahalanobis       |  (x - y)' * Q * (x - y)  |
|  WeightedEuclidean   |  sqrt(sum((x - y).^2 .* w))  |
|  WeightedSqEuclidean |  sum((x - y).^2 .* w)  |
|  WeightedCityblock   |  sum(abs(x - y) .* w)  |
|  WeightedMinkowski   |  sum(abs(x - y).^p .* w) ^ (1/p)  |
|  WeightedHamming     |  sum((x .!= y) .* w)  |
  
**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way.






