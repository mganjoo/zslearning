function [cost, grad] = mapOneShotCostNoAutoenc( theta, data, params )

[ W, b ] = stack2param(theta, params.decodeInfo);

data.imgs = [data.imgs data.zeroimgs(:, 1)];
data.zeroimgs = data.zeroimgs(:, 2:end);
data.categories = [data.categories data.zerocategories(:, 1)];

numImages = size(data.imgs, 2);

% Feedforward
h = bsxfun(@plus, W{1} * data.imgs, b{1});
w = data.wordTable(:, data.categories);

cost = 0.5 / numImages * (sum(sum((h(:,1:end-1) - w(:,1:end-1)).^2)) + params.oneShotMult*sum((h(:,end) - w(:,end)).^2));

% Backprop
del = [h(:,1:end-1) - w(:,1:end-1), params.oneShotMult*(h(:,end) - w(:,end))] / numImages;
Wgrad = del * data.imgs';
bgrad = sum(del, 2);

grad = [ Wgrad(:); bgrad(:) ];

if params.doEvaluate == true
    doPrint = true;
    mapDoEvaluate(data.imgs(:,1:end-1), data.categories(:,1:end-1), data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.validImgs, data.validCategories, data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.testImgs, data.testCategories, data.testOriginalCategoryNames, data.testCategoryNames, data.testWordTable, theta, params, doPrint);
end

end
