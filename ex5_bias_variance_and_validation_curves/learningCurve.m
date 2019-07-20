function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
m_cv = size(Xval,1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% optional: choose random i examples for each i, up to say 10-50 times and
% calculate errors from those random i examples.  Take average of all of those
% and store into error_train and error_val

% doing so will give a more accurate representation (numerical evaluation)
% of the training algorithm depending on the number of training examples

% number of iterations to repeat for each i
repeat_iters = 10
training_errors_per_iter = zeros(repeat_iters,1);
cv_errors_per_iter = zeros(repeat_iters,1);

for i=1:m,
  training_errors_per_iter = zeros(repeat_iters,1);
  cv_errors_per_iter = zeros(repeat_iters,1);
  
  for iter= 1:repeat_iters,
    %choose i random indices from 1:m 
    idx = randperm(m, i);
    X_rand = X(idx,:);
    y_rand = y(idx,:);
    
    theta = trainLinearReg(X_rand, y_rand, lambda);
    
    % store the errors associated to this trained model
    % in training_errors_per_iter and cv_errors_per_iter
    training_errors_per_iter(iter) = norm(X_rand*theta - y_rand,2)^2;
    cv_errors_per_iter(iter) = norm(Xval*theta-yval,2)^2;
  endfor

  % divide each error by 2 times the number of examples used to calculate
  % that particular error; for training_error, this depends on the number
  % of training examples: specifically i.  for cv_error, it is always
  % the size of the cross validation set: specfiically m_cv
  training_errors_per_iter = training_errors_per_iter / (2*i);
  cv_errors_per_iter = cv_errors_per_iter / (2*m_cv);
  
  % take average of all the errors calculated from all iteration of
  % models trained to get accurate representation of error
  error_train(i) = (1/repeat_iters) * sum(training_errors_per_iter);
  error_val(i) = (1/repeat_iters) * sum(training_errors_per_iter);

endfor

% The following is code for training the model based on the first
% i examples, rather than choosing i random examples
%for i=1:m,
  % train model for theta using a subset of X,y (up to ith example)
  %theta = trainLinearReg(X(1:i,:), y(1:i,:), lambda);
  
  %calculate training error using the training subset
  %error_train(i) = (1/(2*i)) * norm(X(1:i,:)*theta - y(1:i),2)^2;
  
  %calculate the cv error using the entire Xval,yval
  %error_val(i) = (1/(2*m_cv)) * norm(Xval*theta - yval,2)^2;
%endfor
% ---------------------- Sample Solution ----------------------







% -------------------------------------------------------------

% =========================================================================

end
