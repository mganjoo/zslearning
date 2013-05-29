function [cost,grad] = softmaxCostMinimizeSeenChange(theta, data, params)

W = stack2param(theta, params.decodeInfo);
numCategories = size(W{1}, 1);

% We define params.outputsToChange as the rows of the weight matrix for
% which we change a lot (for the others we minimize change).
% params.thetaOld are the old parameters.
Wold = stack2param(params.thetaOld, params.decodeInfo);
maskChange = zeros(numCategories, 1);
maskChange(params.outputsToChange) = 1;
maskFix = ~maskChange;

pred = exp(W{1}*data.imgs); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
m = length(data.categories);
truth_inds = sub2ind(size(pred),data.categories,1:m);

WdiffFix = bsxfun(@times, W{1} - Wold{1}, maskFix);
W{1} = bsxfun(@times, W{1}, maskChange);

cost = -sum(log(pred(truth_inds)))/m + (params.lambdaOld/2)*sum(sum(WdiffFix.^2))+(params.lambdaNew/2)*sum(sum(W{1}.^2));

truth = zeros(size(pred));
truth(truth_inds) = 1;
error = pred - truth;
Wgrad = (error*data.imgs')/m + params.lambdaNew*W{1} + params.lambdaOld*WdiffFix;

grad = Wgrad(:);

end