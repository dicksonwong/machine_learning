function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%

% Some notes:
% size(X,2) returns the second entry of size(X), which is rows x cols
% that is, size(X,2) is the number of columns - number of features
% so mu is a 1 row x (n+1) vector
% and similarly sigma is  1 row x (n+1) vector

mu = mean(X);

% OPT=1 normalizes with N rather than N-1, which is the default and also
% OPT=0 argument
sigma = std(X);

% Because mu and sigma are not the same sizes as X, then we need
% to adjust it a little bit.  We can construct a matrix that has
% the same entry along all rows by multiplying it by the column
% vector ones(size(X,2)), as mu is just a row vector.
X_norm = (X .- (ones(size(X,1),1)*mu)) ./ (ones(size(X,1),1)*sigma)


% ============================================================

end
