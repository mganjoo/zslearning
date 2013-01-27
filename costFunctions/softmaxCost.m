function [cost,grad] = softmaxCost(theta, data, params)

W = stack2param(theta, params.decodeInfo);

pred = exp(W{1}*data.imgs); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
m = length(data.categories);
truth_inds = sub2ind(size(pred),data.categories,1:m);
cost = -sum(log(pred(truth_inds)))/m + (params.lambda/2)*sum(sum(W{1}.^2));

truth = zeros(size(pred));
truth(truth_inds) = 1;
error = pred - truth;
Wgrad = (error*data.imgs')/m + params.lambda*W{1};

grad = Wgrad(:);

end