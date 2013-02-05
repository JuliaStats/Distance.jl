# Unit tests for Distances

using Test
using Metrics

# test individual metrics

x = [4., 5., 6., 7.]
y = [3., 9., 8., 1.]

@test sqeuclidean(x, y) == 57.
@test euclidean(x, y) = sqrt(57.)
@test cityblock(x, y) = 13.
@test chebyshev(x, y) = 6.
@test minkowski(x, y, 2) = sqrt(57.)



