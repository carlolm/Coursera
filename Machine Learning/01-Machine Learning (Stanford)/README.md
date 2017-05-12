# Resources

- [Hole House Notes](http://www.holehouse.org/mlclass/index.html)

---

# Week 3

# Classification
- linear regression not suitable since classification is not a linear function

## Logistic Regression
- 0 <= h_{Ɵ}(x) <= 1 
- classification algorithm
- h_Ɵ(x) = g(Ɵ^{T}x)
- **Sigmoid function / logistic function**: g^{z} = 1 / (1 + e^{-z})
 - **h_{Ɵ}(x) = 1 / (1 + e^{-Ɵ^{T}x})**
 - **h_{Ɵ}(x) = p(y = 1 | x;Ɵ)**:
   probability of y = 1 given x is paramaterized by Ɵ
   - h_{Ɵ}(x) >= 0.5 => y = 1
   - h_{Ɵ}(x) < 0.5 => y = 1

 ```
 z = 0, e^0 = 1 ⇒ g(z)=1/2
 z → ∞, e^{−∞} → 0 ⇒ g(z) = 1
 z → −∞, e^∞ → ∞ ⇒ g(z)=0
 ```

 ```
 h_{θ}(x) = g(θ^{T}x) ≥ 0.5 when θ^{T}x ≥ 0
 ```

 ```
 θ^{T}x ≥ 0 ⇒ y = 1
 θ^{T}x < 0 ⇒ y = 0
 ```

- **Decision boundary**: separates region predicted to be y = 1 vs y = 0

## Cost Function
- Cost(h_{Ɵ}(x), y) =
 - -log(h_{Ɵ}(x)) | if y = 1
 - -log(1 - h_{Ɵ}(x)) | if y = 0
 - => Cost(h\_{Ɵ}(x), y) = -y * log*h\_{Ɵ}(x) - (1-y)(log(1-h_{Ɵ}(x)))

```
J(Ɵ) = 1/m sum\_{i=1}^m (Cost(h_{Ɵ}(x^(i)), y^(i)))
```

## Gradient Descent
- Alternative optimization algorithms:
 - (+) no need to manually pick α
 - (+) often faster than gradient descent
   1. Conjugate gradient
   2. BFGS
   3. L-BFGS

## Multiclass Classification

- **One-vs-all (one-vs.rest)**:
 - Change problems to separate binary problems: current target class vs. others
 - Traing a logistic regression classifier to predict probability y = i
 
 ```
 h_{Ɵ}^(i)(x) = P(y=i | x; Ɵ)
 ```

## Summary: Logistic Regression

1. Linear regression is not suitable for classification problems; instead an alternative function - Sigmoid function / logistic function must - is more suitable.

2. Decision boundaries separate the regions predicted as y = i for different values of i and may be linear or non-linear.

3. Cost function for logistic regression: a "non-convex" cost function with multiple local optima must be modified to a "convex" function with a global optimum in order to ensure gradient descent convergence.

4. There exist advanced optimal algorithms to gradient descent, such as (a) conjugate gradient, (b) BFGS, and (c) L-BFGS, which are often faster, do not require α selection, but are more complex.

5. Multiclass classification can be decomposed into a series of binary classifications (one-vs-all), training a logistic regression classifier to predict each case of y = i, versus an aggregate set of all other values of i.

---

## Regularization

- **Overfitting**: high variance
 - Function order too high and lines forced to fit training set, so much so that it results in lots of curves, turns in the function
 - Too many features, may fit training set well but fails to generalize to new examples
- **Underfitting**: high bias, hypothesis maps poorly to the trend of the data, usually due to overly simple function or too few features

### Addressing Overfitting
1. Reduce numer of features
 - Manual selection
 - Model selection algorithm
2. **Regularization**
 - Keep all features, but reduce magnitude/values of parameters Ɵ_{j}

- J(Ɵ) = [cost function] + lambda * sum_{j=1}^n(Ɵ_{j}²)
 - _lambda_: regularization parameter
 - note: lambda summation starts from 1 and not 0

### Regularized Linear Regression

---

# Week 4: Neural Networks
- Neurons are bascially computational units that take inputs (dendrites) as electrical inputs ("spikes") that are channeled to outputs (axons)
- dendrites => input features x_{1}..x_{n}
- outputs => results of hypothesis function
- x_{0} = bias node = 1
- same logistic function as classification:
 - "sigmoid (logistic) activation function": 1 / (1 + e^{-Ɵ^{T}x})
- input nodes (layer 1) = "input layer"
- go into another node (layer 2), which outputs to the hypothesis function "output layer"
- "hidden layers" intermediate layers of nodes between the input and output layers

![](images/activation_layers.png?raw=true)

- a_{i}^(j) = _activation_ of unit i in layer j
- Ɵ^(j) = matrix of weights controlling function mapping from layer j to layer j + 1
- if a network has s\_{j} units in layer j, s\_{j+1} units in layer j + 1, then Ɵ^(j) will have dimensions s\_{j+1} x (s\_{j} + 1)

- **forward propogation**: vectorized implementation

![](images/model_rep2.png?raw=true)

---

- x1 **XOR** x2: only one of x1 or x2 is true
- x1 **XNOR** x2: both are true or both are false

---

# Week 5: Neural Network Cost Function and Backpropagation

- `L` = total number of layers in the network
- `s_{l}` = number of units (not counting bias unit) in layer l
- `K` = number of output units/classes
- **Neural networks cost function:**
 ![](images/NNcostFunction.png?raw=true)

- **Ɵ_{ji}^l**

 ![](images/theta_notation.png?raw=true)
 - `j` (first of two subscript numbers)= ranges from 1 to the number of units in layer l+1
 - `i` (second of two subscript numbers) = ranges from 0 to the number of units in layer l
 - `l` is the layer you're moving FROM

## Backpropagation Algorithm
```
Minimizing cost function for a neural-network.
Objective: minimimize J(Ɵ) w.r.t. Ɵ
```

- Calculate delta_{j}^(l): _error_ of node j in layer l
- delta\_{j}^(l) = a\_{j}^(l) - y\_{j} = h(Ɵ^(x))\_{j} - y\_{j}
- delta^(3) = (Ɵ^(3))^T * delta^(4) .* g'*(z^(3))
 - g'*(z^(3)) = a^(3) .* (1-a^(3))
- Set Delta\_{ij}^(l) = 0 (for all l,i,j)
 - These Delta values will be used to computer the partial derivative, accumulators for computing the partial derivatives


### Methodology
1. Set a^(1) := x^(t)
2. Perform forward progapation to compute a^(l) for l = 2,3,...,L
3. Using y^(t), computer delta^(L) = a^(L) - y^(t)
4. Compute δ^(L−1), δ^(L−2),…, δ^(2) using δ^(l)=((Θ^(l))^T * δ^(l+1)) .∗ a^(l) .∗ (1−a^(l))
5. Δ^(l)\_{i,j} := Δ^(l)\_{i,j}+a^(l)\_{j} * δ^(l+1)\_{i} or with vectorization, Δ^(l) := Δ^(l) + δ^(l+1) * (a^(l))^T
 - D^(l)\_{i,j} := (1/m)(Δ^(l)\_{i,j} + λΘ^(l)\_{i,j}), if j≠0.
 - D^(l)\_{i,j} := (1/m)Δ^(l)\_{i,j} If j=0

![](images/back_prop.png?raw=true)

- unrolling matrices into vectors in Octave:
 ```
 ƟVec[ Θ1(:); Θ2(:); Θ3(:)];
 ```
- extracting original matrices:
 ```
 reshape(ƟVec(<start>:<end>), <rows>, <cols>)
 ```

### Gradient Checking

= Numeric method (compare derivate versus a calculated slope between two points) to ensure implementation is working, i.e. converging
 ```
 gradApprox = (J(Ɵ + ε) - J(Ɵ - ε))/(2*ε)
 ```
- Approximating for each Ɵ_{i}:
 ```
 EPSILON = 1e-4;
 for i = 1:n,
 	ƟPlus = Ɵ;
 	ƟPlus(i) = ƟPlus(i) + EPSILON;
 	ƟMinus = Ɵ;
 	tehtaMinus(i) = ƟMinus(i) - EPSILON;
 	gradApprox(i) = (J(ƟPlus)-J(ƟMinus)) / (2 * EPSILON);
 end;

 % Compare gradApprox ~ DVec, DVec = derivatives from backprop
 ```

- **Implementation Summary**

 1. Implement back propagation to compute DVec
 2. Implement numerical gradient checking to compute gradApprox
 3. Check they're basically the same (up to a few decimal places)
 4. Before using the code for learning turn off gradient checking
 	- GradAprox stuff is very computationally expensive
 	- In contrast backprop is much more efficient (just more fiddly)

### Random Initiation

- Initializing as zero fails (unlike linear regression which works), makes all activation layers the same
- Having all of the same values for Ɵ creates symmetry and may result in calculation of activation layers all being the same
- Use random values (between 0 and 1, scaled by ε): ε * rand
	```
	Θ1 = rand(10,11) * (2*INIT\_ε) - INIT\_ε;
	Θ2 = rand(10,11) * (2*INIT\_ε) - INIT\_ε;
	% rand(10,11) = generates 10xp11 matrix of random (0,1)

	Θ3 = rand(1,11) * (2*INIT\_ε) - INIT\_ε;
	```

### Putting It All Together

- Number of input units: dimension of features x^i
- Number of output units: number of classes
- Default: 1 hidden layers
- > 1 hidden layer: same number of nodes per hidden layer
- The more hidden units, the better; however, more computationally expensive

---

**Training a Neural Network**
1. Randomly initialize weights
2. Implement forward propagation: get h_{Ɵ}(x^(i)) for any x^(i)
3. Implement code to compute cost function J(Ɵ)
4. Implement backprop to compute partial derivatives ∂/∂Ɵ^(l)_{jk} J(Ɵ)
	```
	for i = 1:m {
		Forward propagation on (x^i, y^i) --> get activation (a) terms
		Back propagation on (x^i, y^i) --> get delta (δ) terms
		(a(l) and δ(l) for l = 2,...,L)
		Compute Δ := Δ^l + δ^(l+1)(a^l)^T
	}

	% With this done compute the partial derivative terms
	```
5. Gradient checking to compare partial derivs vs. numerical estimates of gradient J(Ɵ)
6. Gradient descent or advanced optimization method with backprop to minimimze J(Ɵ) as a function of Ɵ

### Week 5 Summary

1. A δ term on a unit as the "error" of cost for the activation value associated with a unit, in order to compare the output of the neural network vs. desired output.
2. Neural network architecture: (a) # of input units = number of features, (b) # of output units = number of classes, (c) hidden layers should have the same number of nodes, (d) the more hidden units the better, however is more computationally expensive.
3. Gradient checking: numeric method to ensure implementation is working (gradient descent is converging) by manually calculating the approximate slope between two points (point +/- a margin ε) and comparing it with the partial derivative.
4. Initial Ɵs should be different/randomized (between 0 and 1, scaled by ε) to avoid symmetry and ensure activation layers are calculated to be different values.
5. Steps for training a neural network: (a) randomly initialize weights/Ɵs, (b) implement forward propagation to calculate h_{Ɵ}(x^(i)) for any x^(i), (c) compute cost function J(Ɵ), (d) back propagation to compute partial derivatives/delta terms, (e) gradient checking, (f) gradient descent or advanced optimization method to minimize J(Ɵ) as function of Ɵ.

---

# Week 6: Advice for Applying Machine Learning

- Ways to improve results:
	1. More training examples
	2. Smaller/larger sets of features
	3. Adding polynomial features
	4. Decreasing/increasing lambda
- **Machine learning diagnostic**: test to gain insight on what's working/not working
- **Method 1: Split Data Set**
	- 70% training set
	- 30% test set: m_{test}
	- (1) Learn parameter Ɵ from training set (min J(Ɵ))
	- (2) Compute test set error: J_{test}(Ɵ)
- Misclassification error:
	- err(h_{Ɵ}^(x), y) =
		- | 1 <= h(x) >= 0.5, y = 0 or h(x) < 0.5, y = 1
		- | 0 <= otherwise
	- test error = 1/m\_{test} * sum[i..m_test] (h(x\_test) - y^i)

	![](images/test_error.png?raw=true)
- **Improved Model Selection**
	- 60% training | 20% cross validation | 20% test set
	- J_{x}(Θ): x = training error | cv error | test error
	- (1) Optimize parameters in Ɵ using the training set for each polynomial degree
	- (2) Find the polynomial with the least error using the cross validation set
	- (3) Estimate the generalization error using the test set with J_{test}(Ɵ^(d))

### Diagnosing Bias vs. Variance

![](images/bias_variance.png?raw=true)

- **High bias**: underfitting problem
	- Training error (J train Ɵ) = high
	- CV error = similar to J train
- **High variance**: over fitting problem
	- Training error (J train Ɵ) = low
	- CV error >> training error
- Higher degree polynomial: (1) lower training erorr, (2) cross validation error quadratic: will decrease initially and then increase at a certain point


### Regularization

1. Choose several values of lambda (typically increment by 2x)
2. Create models with different degrees and other variants
3. Train several models with different lambdas, by minimizing J(Θ) => Θ^(p)
4. Use CV set to validate the hypotheses => CV error
5. Pick the model that gives the lowest error
6. Use the selected model on the test set
- Plot J(train) / J(cv)

### Learning Curves

- Plot J(train) / J(cv) as a function of m (training set size)
- Training size small => error will be small (easier to fit)
	- => (1) training set error increases with m
	- => (2) J(cv) decreases with m

**High Bias**
- (1) and (2) converge, but (2) levels off
- Additional training data will not help
=> Models are typically not complex enough for the data and tend to underfit

**High Variance**
- (1) and (2) converge but have a gap
- (2) does not level off but gap persists
- getting more training data will help
=> Complex models which tend to overfit

### Debugging a Learning Algorithm
1. Get more training examples => fixes high variance
	- not good for high bias
2. Try a smaller set of features => fixes high variance
	- not good for high bias
3. Get additional features => fixes high bias (because hypothesis is too simple)
4. Try adding polynomial features => fixes high bias
5. Try decreasing lambda => fixes high bias
6. Try increasing lambda => fixes high variance

**Neural Networks and Overfitting**
- "Small" NN: fewer parameters, more prone to underfitting
	- computationally cheaper
- "Large" NN: more parameters, more prone to overfitting
	- computationally more expensive
	- use regularization (lambda) to address overfitting
- Increasing number of hidden units does not help high variance

**Model Complexity Effects**
- Lower-order polynomials/low model complexity: high bias, low variance.  Typicaly poor fitting
- Higher-order polynomials/high model complexity: 

### Week 6 Summary

1. Ways to improve results: (1) more training examples, (2) smaller/larger sets of features, (3) adding polynomial features, (4) decreasing/increasing lambda
2. Splitting data into sub sets (60% training | 20% cross validation | 20% test set) allows for better diagnosis of hypothesis.
3. Improved model selection can be achieved by (1) training different polynomial hypothesis expressions on the training set, (2) selecting the expression with lowest error on the cross validation set, and (3) estimating the generalization of the selected model on the test set.
4. High bias typically results from underfitting and an overly simple model, while high variance typically results from overfitting and a model that is too specifically derived from the training set.
5. For regularization, (1) choose range of lambda values, typically incremented by a factor of 2, (2) create models with different degrees and other variants, (3) train several models with different lambdas, (4) use the cross validation set error to choose the model, (5) use the selected model on the test set, (6) plot training error vs. cross validation error
6. Smaller, simpler neural networks with fewer parameters are prone to underfitting, while larger neural networks with more parameters are prone to overfitting.
7. Learning curves are plots of cost function / J(Θ) for the training set and the cross validation set vs. the training set size and can be used to analyze performance and diagnose problems.
8. For high variance, try (1) more training examples, (2) smaller feature set, (3) increasing lambda, while for high bias, try (1) additional features, (2) adding polynomial features, (3) decreasing lambda.

### Error Analysis

**Recommended approach**
1. Start with simple algorithm that can be quickly implemented
2. Plot learning curves: diagnose bias/variance and modify algorithm accordingly
3. Error analysis: manualy examine the examples (in CV) that the algorithm made errors on to spot trends/identify source of error

- Get a numerical value (for example 3% vs 5% error rate) to evaluate whether or not the feature materially improves performance

**Error Metrics for Skewed Classes**
- **skewed classes**: classification where one case may be materially higher than the opposit case (e.g. 1% cancer vs. 99% no cancer)
	- example problem: predicting no cancer for all cases has low error!

![](images/error_skewedClasses.png?raw=true)

- **precision**: % of correct positives vs. total predicted positives
	- true positives / # predicted as positive = true positives / (true positives + false positives)
- **recall**: % of correct positives vs. total actual positives
	- true positives / # of actual positives = true positives / (true positives + false negatives)
- Trading off precision and recall:
	- Predict 1 if h(x) >= (0.5) => 0.7+
	- Predict 0 if h(x) < (0.5) => 0.7+
	- Increase threshold / higher confidence for positive case
	- => results in higher precision / lower recall
- Optimizing precision/recall: **F₁ Score** = 2 * P * R / (P + R)

### Large data rationale
- Training error will be small, large training set will reduce variance

---

# Week 7: Support Vector Machines
- Logistic regression formula re-written:
	![](images/SVM_costFunc.jpg?raw=true)

![](images/SVM.png?raw=true)
- Minimizing the first term:
	- y^(i)=1 => Θ^T * x^(i) > 1 (not just > 0)
	- y^(i)=0 => Θ^T * x^(i) < -1 (not just < 0)
- **SVM decision boundary**
	- **Large margin classifier**: try to create a boundary / cushion between classes 

### Large Margin Classifiers
- **Vector inner product**:
	- 2 vectors u and v, where u = [u1; u2] and v = [v1; v2]
	- inner product = u^T v
	- **norm** = **||u||**: of vector is the length of the vector (hypotenuse: sqrt(u1² + u2²))

## Kernels
- Given x, compute new feature depending on proximity to landmarks
- Choosing landmarks: use same locations as training examples 

### SVM parameters
- C = 1 / lambda:
	- large C == small lambda: lower bias, high variance => more prone to overfitting
	- small C == large lambda: higher bias, low variance => more prone to underfitting
- σ²:
	- large: features f_{i} vary more smoothly: decreased change
		- higher bias, lower variance / underfitting
	- small: vary less smoothly
		- lower bias, higher variance / overfitting

### Methodology
1. Use SVM software to solve for parameters Θ
2. Choice of parameter C
3. Choice of kernel (similarity function)
	- No kernel ("linear kernel): predict "y=1" if Θ^T * x >= 0
	- Gaussian kernel:
	![](images/gaussian_kernel.png?raw=true)

---

- Implementation of kernel function:
	```
	function f = kernel(x1,x2)
	f = f_i
	x1 = x^(i)
	x2 = l^(j) = x^(j)
	```
- **polynomial kernel**: k(x,l) = (x^T * l)²
	- less used
	- (x^T * l + constant)^degree
	- (x^T \* l)² | (x^T \* l + 1)³ | (x^T \* l + 5)⁴
- **other kernal functions**: string kernel, chi-squared kernel, histogram intersection kernel

### Multi-Class Classification
- Train K SVMs using one-vs-all method
- Pick class i with largest (Θ^i)^T * x

### Logistic Regression vs. SVM
- if n is large (relative to m): use logistic regression or SVM without a kernel ("linear kernel")
	- n = 10,000 | m = 10...1,000
- if n is moderate: use SVM or Gaussian kernel
	- n = 1-1,000 | m = 10-10,000
- if n is small, m is large: create/add more features and use l/r or SVM w/o kernel

### Spam Filter
1. Preprocess emails: e.g. lower casing, stripping HTML, normalizing URLs/email addresses/numbers/dollars, word stemming (discounts|discouted|discounting => discount), removal of white spaces/punctuations
2. Build a vocabulary list and prioritize most freqently occurring

---

# Week 8: Clustering
- Unsupervised learning: 

## K-means algorithm
1. Randomly allocate (two) points as the **cluster centroids** 
2. Cluster assignment
	- assign each example to the closest centroid
3. Move centroid step
	- move each centroid to the average of the correspondingly assigned data-points

---
- c^(i) = index of cluster x^(i) 
- μ_k = cluster centroid
- μ_{c^(i)} = cluster centroid of cluster to which example x^i has been assigned

### Optimization objective
- min {c^i,...,c^m | μ\_1,...,μ_K} J() = 1/m Σ\_{i=1...m} || x^i - μ\_{c^(i)} ||²

### Random initialization
- should have K < m
- randomly pick < training examples
- set μ_1,...,μ_k equal to these K examples
- => randomly choose 2 examples as your 2 centroids
- potential for getting stuck at local optima
	- mitigation: run several different initializations
	- typically 50 - 1,000 times
	- compute cost function
	- choose clustering with lowest cost

### Choosing number of clusters (K)
- typically by looking at data; no clearly defined methodology
- **elbow method**: plot cost function J vs. K
	- choose inflection point of the curve
	- rarely used since data usually does not produce a clear "elbow"

## Data Compression

## Dimensionality Reduction
- 2D to 1D: project all points onto a close fitting line
	- `x^i in R^2 => z^i in R^1`
- 3D to 2D: project onto a close fitting 2D plane
	- `x^i in R^3 => z^i in R^2`
- typically reduce to 1D/2D/3D to allow for vizualization

### Principal Component Analysis (PCA)
- Tries to define a surface for projection that minimizes projection error (distance from points to projection)
- find k vectors u^1,...,u^k onto which to project the data
- **PCA is not linear regression**: 

### PCA Algorithm
- Data preprocessing: [feature scaling] / mean normalization
- Compute **covariance matrix**:
	- Σ = 1/m sum[i=1...n] (x^i)(x^i)^T
	- n x n matrix: x^i = n x 1 | (x^i)^T = 1 x n
	- => Σ = n x n
- Computer **eigenvectors** of matrix Σ
	- singular value decomposition
	- Octave: `[U,S,V] = svd(Sigma)` | `eig(Sigma)`
	- U = n x n matrix: columns u^1,...,u^n
	- U in R^(n x n)
	- Use first k columns to get u^k => n x k matrix = U_{reduce}
	- z = U_{reduce}^T * x
	- Σ = 1 / m * X' * X

```
[U,S,V] = svd(Sigma);
Ureduce = U(:, 1:k);
z = Ureduce' * x;
```

- S: same "shape" as identity matrix

### Reconstruction from Compressed Representation
- X\_approx [n x 1] = U\_reduce [n x k] * z [k x 1]

### Choosing k
- average squared projection error / total variation in the data <= 0.01
- 99% variance retained

```
(1/m Σ\_{i=1...m} || x^i - μ\_{c^(i)} ||²) / (1/m Σ\_{i=1...m} || x^i ||²) <= 0.1 (1%)
```

### Supervised learning speedup
- (x^1, y^1)...(x^m, y^m)
- Extract inputs => x^1,...,x^m 
- PCA => z^1,...,z^m
	- Run PCA only on training data
- new training set: (z^1,y^1),...,(z^m,y^m)

---

- PCA is not a good way to address overfitting; instead use regularization
- Try analyzing existing data first (x^i); only use PCA if using existing raw data does not yield intended results

# Week 9: Anomaly Detection
- Applications: fraud detection (unusual behavior), monitoring computers in a data center
- anomaly if p(x) <= epsilon or p(x) >= epsilon

## Gaussian (Normal) distribution
- if x is a distributed Guassian with mean mu, variance σ^2:
	- x ~ 𝒩(μ, sigma^2)
- P(x:μ,σ²) = probability of x paramagterized by mean and squared variance:
	- 1 / sqrt(2 * pi * σ) * exp(-(x-μ)/2σ²)

## Algorithm Evaluation
1. Fit model p(x) on training set {x^1...x^m}
2. On CV/test example x, predict:
	- y = 1 if p(x) < epsilon (anomaly)
	- y = 0 if p(x) >= epsilon (normal)

Anomaly Detection | Supervised Learning
---|---
Small number of positive examples, large number of negative examples | Large number of positive/negative examples
Many different "types" of anomalies | Enough positive examples to get a sense of what +ve examples are like

### Examples
Anomaly Detection | Supervised Learning
---|---
Fraud detection | Email spam classification
Manufacgturing | Weather prediction
Monitoring machines in data center | Cancer classification

## Features:
- Use transformation to get data to have a Gaussian distribution
	-	log(x) | log(x + c) | x^(1/2) | x^(1/3)
- Choose features that might take on unusually large or small values in the event of an anomaly

## Multivariate Gaussian (Normal) Distributions

```
det(Sigma) // compute determinant of matrix Sigma = | Sigma |
```

- smaller Sigma => narrower Gaussian distribution

![](images/gaussian_ex.png?raw=true)

- negative values => negative correlation
- mu ∈ R^n | Sigma ∈ R^(n x n)

![](images/probabilityFormula.png?raw=true)

## Methodology
1. Fit model p(x) by setting mu & Sigma
2. Compute p(x) for new example x

## Recommender Systems
- n\_u: # of users | n\_m: # of movies | r(i,j) = 1 if user j has rated movie i | y^(i,j) = rating given by user j to movie i

![](images/optimizationAlgo.png?raw=true)

### Collaborative filtering 
1. Initialize x1,...,x^(n\_m), Θ1,...,Θ^(n\_u)
2. Minimize J() using gradient descent
3. For a user with parameters Θ and a movie with (learned) features x, predict a start rating of Θ^T * x

### Mean Normalization
- (a) Average all ratings, (b) subtract mean from all ratings

---

# Week 10: Large Scale Machine Learning
- **Check if algorithm improves with a larger data set**: Plotting a learning curve for a range of values for m: verify that the algorithm has high variance when m is small
	- If no improvement, can consider adding extra features / hidden nodes

## Stochastic Gradient Descent
- Scales better, faster
- Cost function may not necessarily go down with every iteration (as with batch gradient descent)

**Methodology**
1. Randomly shuffle (reorder) training examples
2. Repeat for i = 1 ... m:
	- Θj := Θj - α (h_Θ (x^i) - y^i) * x\_j^i
	- For every j = 0,...,n

---
- α is typically held constant.
- Can slowly decrease α over time if we want Θ to converge (e.g. α = const1 / (iterationNumber + const2))

## Mini-Batch Gradient Descent

Algorithm | Description
----------|------------
Batch gradient descent | Use all m examples in each iteration. Converges to the global minimum
Stochastic gradient descent | Use 1 example in each iteration. Generally moves in the right direction and gets near the global minimum, but may not converge
Mini-batch gradient descent | Use b examples in each iteration

- b: mini-batch size, 10 / 2-100
- For-loop through b-size batches of m
- Allows for vectorized implementation
- Can partially parallelize computation (e.g. do 10 at once)
- (-) optimization of parameter b

# Week 11: Application Example - Photo OCR

- **Pipelines**: separate modules / stages of solving a problem, each of which may involve ML
- Synthesizing data: creating data samples to improve learning, e.g. text images / sound samples with distortion/noise
- Ceiling analysis: measure potential improvement in accuracy for each module of the pipeline in order to prioritize whcih components to work on