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






