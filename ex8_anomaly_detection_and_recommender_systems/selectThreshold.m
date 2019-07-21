function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
    % Use logical array to get the predictions
    predictions = (pval < epsilon);
    
    % total number of true positives + false positives is
    % equal to number of 1s in predictions
    positives = sum(predictions);
    
    % number of true positives is the number of times 1 appears in
    % predictions & yval; implicitly, a logical AND returns 1 iff
    % both prediction and actual value is equal to 1
    true_positives = sum(predictions & yval);
    
    % precision is true positives / positives; if the number
    % of positives is equal to 0, let us define precision = 0
    if positives > 0
      precision = true_positives / positives;
    else
      precision = 0;
    endif
    
    % false negative occurs when we predict 0 but actual is 1;
    % by taking xor of yval and predictions, we determine the
    % predictions that were incorrect; then we take logical 
    % AND with yval to see where we predicted incorrectly AND
    % the actual value was 1 - hence, where we inccorrectly 
    % predicted 0 (or the false negatives)
    false_negatives = sum(xor(yval,predictions) & yval);
    
    % recall is true positives / (tp + fn); if tp + fn == 0, then
    % define recall = 0
    if true_positives > 0 || false_negatives > 0
      recall = true_positives / (true_positives + false_negatives);
    else
      recall = 0;
    endif

    % F1-score is 2*prec*rec/(prec+rec) only if well-defined
    if recall > 0 || precision > 0
      F1 = (2 * precision * recall) / (precision + recall);
    else
      F1 = 0;
    endif

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end

end

end
