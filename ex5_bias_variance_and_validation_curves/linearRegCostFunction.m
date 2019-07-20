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

% calculate non-regularized cost function for linear regression
J_nonreg = (1 / (2*m)) * norm(X*theta-y, 2)^2;

% define theta_reg to be equal to theta, except with the
# first row equal to 0; the term theta_0 does not get regularized
theta_reg = [0; theta(2:end)];

% regularized cost function for linear regression
J = J_nonreg + (lambda / (2 * m)) * (norm(theta_reg, 2)^2);

% gradient for regularized linear regression
grad = (1/m)* (X' * (X*theta - y)) + (lambda/m) * theta_reg;






% =========================================================================

grad = grad(:);

end
