function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% First, calculate the sum of the squared errors; the (i,j)th term
% in the product x*theta' is exactly equal to the prediction of
% movie rating for movie i and user j; we need only sum up all
% terms where the corresponding (i,j)th value in the matrix R is
% equal to 1; hence, subtract Y from X*Theta', take entry-wise 
% with R, square each term, then sum all terms

% For brevity in the next few parts, we will define the matrix
% containing such error terms for movies that have been rated
% errors_def (for errors that have been defined)
errors_def = (X*Theta' - Y) .* R;

% Non-regularlized cost
J_nonreg = (1/2) * sum(sum(errors_def.^2));

% Calculate the gradient for X; note that the (i,k) term in the
% matrix X_grad is equal to the sum over all terms in row i
% (coresponding to movie i) in the matrix (X*Theta' - Y) .* R 
% multiplied to the corresponding term in the kth feature column
% in Theta.  In fact, we can easily obtain all terms through
% matrix product (X*Theta' - Y) .* R) * Theta, which is n_movies
% * n_features matrix  
X_grad = errors_def * Theta;

% Similarly, calculate the gradient for Theta; note that the (j,k)
% term in this matrix is equal to the sum over all terms in col j
% (corresponding to user j) in matrix errors_def multiplied to the
% cooresponding term in the kth feature row of X.  This also
% has the easy product errors_def' * X, which is n_users by n_features
Theta_grad = errors_def' * X;

% Calculate regularization term for the features X and Theta for cost
X_reg_term = (lambda / 2) * sum(sum(X .^ 2));
Theta_reg_term = (lambda / 2) * sum(sum(Theta .^ 2));
J = J_nonreg + X_reg_term + Theta_reg_term;

% Calculate regularization term for features X and Theta for gradient
% For each (i,k), we just need to add lambda * x_i_k to each corresponding
% term in X_grad; that is, X_grad + lambda * X; similarly, Theta_grad =
% Theta_grad + lambda * Theta
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
