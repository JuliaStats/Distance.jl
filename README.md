# Distances

Evaluation of distances between vectors.

## A List of supported distances

* Euclidean distance
* Squared euclidean distance
* Cityblock distance 
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Spearman distance
* Kullback-Leibler divergence
* Jensen-Shannon divergence

## Features

* Many of the distances above accepts a weight vector as an optional argument to calculate weighted distances.
* The module supports computation of distances in different ways:
	- compute a distance between two vectors
	- compute distances between a vector and an array comprised of multiple vectors
	- compute distances between corresponding vectors in two arrays along a specific dimension
	- compute distances between columns in two matrices in a pairwise manner
* Specialized functions are used to compute pairwise (Squared) Euclidean distances in a much faster way.	
	