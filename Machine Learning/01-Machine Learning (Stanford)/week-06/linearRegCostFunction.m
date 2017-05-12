% ***************************************
% 2 April 2017
% Carlo P. Las Marias | carlol@gmail.com
% ***************************************

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% disp("Size y: "); disp(size(y));
% disp("Size lambda: "); disp(size(lambda));
% disp("m: "); disp(m);
% disp("Size X: "); disp(size(X));
% disp("Size theta: "); disp(size(theta));
% disp("Size h: "); disp(size(h));

%   X = m x n matrix, where:
%       m = size of data set
%       n = number of features of hypothesis 
%   theta = n x 1
%   h = m x 1 vector, predicted value for each sample

h = X * theta;

% linear regression cost function
J = 1/(2*m) * sum((h - y) .^2) + lambda / (2*m) * (theta(2:end)' * theta(2:end));

%   Regularized linear regression gradient
%   Partial derivative of regularized linear regression's cost for theta_j
%   grad = n-sized vector

grad = 1/m * X' * (h - y) + lambda / m * theta;
grad(1) = 1/m * sum(h - y) * X(1);

% disp("Size grad: "); disp(size(grad));

% =========================================================================

grad = grad(:);

end
