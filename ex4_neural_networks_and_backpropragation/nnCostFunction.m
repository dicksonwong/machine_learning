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


% Part 1: forward propragation to calculate cost function

% add a column of ones to X for bias unit
X = [ones(m,1) , X];

% layer 1 feed forward to layer 2: calculate layer 2 activations
% This is slightly less expensive: take g(X*Theta1')'
a_2 = (sigmoid(X*Theta1'))';

% add a row of ones to top row of a_2 for bias unit
a_2 = [ones(1,m);a_2];

a_3 = sigmoid(Theta2 * a_2);


% "unpack" y so that we get a k by m matrix such that
% for every example i=1,2,...m, if y(i) = class(i)
% then the ith column of Y has the element at index class(i)
% equal to 1 and every other element in that column is 0
Y = zeros(num_labels, m);
for i = 1:m
  Y(y(i,1),i) = 1;
endfor

% take sum of Y_(i),k multipled by log(a_3)
% + sum of (1-Y_(i),k multiplied by 1-log(a_3)
J = ((-1)/m)*sum(sum( Y .* log(a_3) + (1-Y) .* log(1-a_3)));



% regularized cost calculation:
% remove the first columns of Theta1 and Theta2, as it includes
% weights associated to bias units;
% square them and then sum up the terms
J = J + (lambda / (2*m)) * ...
      (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end).^ 2)));




% This first implementation we perform will be using a for loop
% over all training examples
for t = 1:m
  % Get row t of X for example indexed t
  x_t = X(t, :)';
  a_1 = x_t;
  
  %logical array - This essentially gets t-th column of Y
  y_t = (1:num_labels == y(t,1))';

  % calculate layer 2 activations
  z_2 = Theta1*x_t;
  a_2 = sigmoid(z_2);
  
  % add a 1 term to the top of a_2 for bias unit
  a_2 = [1;a_2];
 
  % calculate layer 3 activations
  z_3 = Theta2*a_2;
  a_3 = sigmoid(z_3);

  % calculate delta_3, the error of the model defined by the Thetas
  delta_3 = a_3 - y_t;
  
  % calculate delta_2, the "error caused" by each unit in layer 2 
  % in this line, it also includes bias unit
  % delta_2 = ((Theta2)' * delta_3) .* (a_2 .* (1-a_2));
  
  % no bias included here; delta_2 represents how much we should
  % change the neural netwrk weights in order to affect
  % the intermediate values in computation so to affect the
  % final output and cost
  delta_2 = ((Theta2(:,2:end))' * delta_3) .* sigmoidGradient(z_2);
 
  % calculate Theta1_grad and Theta2_grad
  % Theta2_grad = Theta2_grad + delta_3 * a_2';
  % Theta1_grad = Theta1_grad + delta_2(2:end) * a_1';
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  Theta1_grad = Theta1_grad + delta_2 * a_1';
endfor

% divide gradients by number of training examples
Theta2_grad = (1/m)*Theta2_grad;
Theta1_grad = (1/m)*Theta1_grad;

% regularize by adding Theta2 and Theta1 but with the first columns
% replaced with 0 - no regularization associated to bias unit weights
Theta2_reg = [zeros(num_labels,1), Theta2(:,2:end)];
Theta1_reg = [zeros(hidden_layer_size,1), Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * Theta2_reg;
Theta1_grad = Theta1_grad + (lambda / m) * Theta1_reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
