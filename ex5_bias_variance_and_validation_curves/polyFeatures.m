function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
%

% Set the first column of X_poly to X
X_poly(1:end, 1) = X; 

% For all columns between 2 to p, set the column of X_poly to the
% previous column multiplied by X
for i = 2:p,
  X_poly(1:end,i) = X_poly(1:end,i-1) .* X;
endfor





% =========================================================================

end
