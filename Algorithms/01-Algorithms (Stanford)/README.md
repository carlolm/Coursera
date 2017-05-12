# Algorithms Study and Review

- [Coursera: Stanford University Algorithms Course 01](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome) Divide and Conquer, Sorting and Searching, and Randomized Algorithms



# Week 1: Design and Analysis of Algorithms I

## Big-Oh: Formal Definition
- T(n) = O(f(n)) if and only if there exist constants c, n\_0 > 0 such taht T(n) <= c * f(n) for all n >= n\_o
- If T(n) = a\_k * n^k + ... + a\_1 * n + a_0, T(n) = O(n^k)
  - Lower order terms and constants are suppressed
  - Only looking for upper bounds: (1) replacing all coefficients with |a| only increases the total, (2) replacing all factors by the highest only makes the total higher.  Therefore the highest order with some coefficient must represent the upper bound

Bound | Description
----- | -----------
Big O | upper bound
Big Omega (Ω) | lower bound
Big Theta (Θ) | tight bound on time
Little 0 | every constant c > 0 there exists constant n\_0 such that f(x) < c g(x) for all x > n\_0

Bound | Description
----- | -----------
Big O | T(n) <= c * f(n) ∀n >= n_0
Big Omega (Ω) | T(n) >= c * f(n) ∀n >= n_0
Big Theta (Θ) | c0 * f(n) <= T(n) <= c1 * f(n) ∀n >= n_0
Little O | T(n) <= c * f(n) ∀n >= n_0

- *T(n) greater than / less than c * f(n) for all n greater than some constant n_0*

# Week 2: Divide and Conquer Algorithms
- Comparing inversions in an array: a good way to check similarity/dissimilarity of people and their preferences