# Tests of At_Q_B, At_Q_A

using Distance
using Base.Test

# helpers

is_approx(a::Number, b::Number, tol::Number) = abs(a - b) < tol
all_approx(a::Array, b::Array, tol::Number) = size(a) == size(b) && all(abs(a - b) .< tol)

tol = 1.0e-12

# tests

q = rand(5)
a = rand(5)
b = rand(5)
A = rand(5, 3)
B = rand(5, 4)
at = a'
At = A'

Q = diagm(q)

@test is_approx(At_Q_B(q, a, b), dot(a, Q * b), tol)
@test all_approx(At_Q_B(q, a, B), vec(at * Q * B), tol)
@test all_approx(At_Q_B(q, A, b), vec(At * Q * b), tol)
@test all_approx(At_Q_B(q, A, B), At * Q * B, tol)

@test is_approx(At_Q_A(q, a), dot(a, Q * a), tol)
@test all_approx(At_Q_A(q, A), At * Q * A, tol)

Q = rand(5, 6)
a = rand(5)
b = rand(6)
A = rand(5, 3)
B = rand(6, 3)
at = a'
At = A'

@test is_approx(At_Q_B(Q, a, b), dot(a, Q * b), tol)
@test all_approx(At_Q_B(Q, a, B), vec(at * Q * B), tol)
@test all_approx(At_Q_B(Q, A, b), vec(At * Q * b), tol)
@test all_approx(At_Q_B(Q, A, B), At * Q * B, tol)

Q = rand(5, 5)

@test is_approx(At_Q_A(Q, a), dot(a, Q * a), tol)
@test all_approx(At_Q_A(Q, A), At * Q * A, tol)

