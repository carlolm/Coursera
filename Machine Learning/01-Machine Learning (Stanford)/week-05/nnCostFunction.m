% ***************************************
% 24 March 2017
% Carlo P. Las Marias | carlol@gmail.com
% ***************************************

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ====== 1.3) Feedforward and cost function

a1 = [ones(m,1) X]; % m x (n+1)
z2 = a1 * Theta1'; % m x (n[2])
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

y_k = zeros(num_labels, m);
for i=1:m,
	y_k(y(i),i) = 1;
end;
J_unreg = 1/m * sum(sum((-y_k .* log(h') - (1-y_k) .* log(1-h'))));

% ====== 1.4) Regularized cost function

reg = lambda / (2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J_unreg + reg;

% ======= 2.3) Backpropagation

% size(Theta1) = n2 x (n1 + 1) = 25 x 401
% size(Theta2) = n3 x (n2 + 1) = 10 x 26

for t = 1:m,

	% (1) forward prop
	a1_i = [1 X(t, :)]; 		% 1 x n1 + 1 = 1 x 401
	z2_i = Theta1 * a1_i'; 	% n2 x 1 = 25 x 1
	a2_i = [1; sigmoid(z2_i)]; 	% (n2 + 1) x 1 = 26 x 1
	z3_i = Theta2 * a2_i; 		% n3 x 1 = 10 x 1
	a3_i = sigmoid(z3_i);		% n3 x 1 = 10 x 1

	% (2) backprop
	d_3 = a3_i - y_k(:,t);

	% (3) 		
	d_2 = (Theta2' * d_3) .* sigmoidGradient([1; z2_i]);

	% delete d_{0}^2
	d_2 = d_2(2:end);

	% (4) Accumulate gradient
	Theta1_grad = Theta1_grad + d_2 * a1_i;
	Theta2_grad = Theta2_grad + d_3 * a2_i';

end;

% (5)
% Theta1_grad = Theta1_grad / m; % 25 x 401
% Theta2_grad = Theta2_grad / m; % 10 x 26

% ======= 2.5) Regularizing Neural Networks

Theta1_grad(:,1) = Theta1_grad(:,1) / m;
Theta2_grad(:,1) = Theta2_grad(:,1) / m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) / m + lambda/m * Theta1(:, 2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) / m + lambda/m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
