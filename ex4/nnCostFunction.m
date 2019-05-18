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

ys = size(y, 1);
Y = zeros(ys, num_labels);

for c = 1:ys
  Y(c, y(c)) = 1;
end

basicCost = (1 / m) * sum(sum((-Y .* log(forwardprop(X, Theta1, Theta2)) - (1 - Y) .* log(1 - forwardprop(X, Theta1, Theta2)))'));
J = basicCost + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

[output_layer, unactivated_hidden_layer_with_bias, hidden_layer_with_bias] = forwardprop(X, Theta1, Theta2);
output_delta = output_layer - Y;

l2_delta = (output_delta * Theta2) .* sigmoidGradient(unactivated_hidden_layer_with_bias);
l2_delta_no_bias = l2_delta(:, 2:end);

Theta2_grad = Theta2_grad + ((1 / m) * (output_delta' * hidden_layer_with_bias));
Theta1_grad = Theta1_grad + ((1 / m) * (l2_delta_no_bias' * [ones(m, 1) X]));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


function [output_layer, unactivated_hidden_layer_with_bias, hidden_layer_with_bias] = forwardprop(X, Theta1, Theta2)
m = size(X, 1);
X_with_bias = [ones(m, 1) X];

unactivated_hidden_layer = X_with_bias * Theta1';
n = size(unactivated_hidden_layer, 1);

unactivated_hidden_layer_with_bias = [ones(n, 1), unactivated_hidden_layer];
hidden_layer_with_bias = [ones(n, 1), sigmoid(unactivated_hidden_layer)];

output_layer = sigmoid(hidden_layer_with_bias * Theta2');
end
