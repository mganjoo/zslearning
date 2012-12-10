function [cost, grad] = mapOneShotCostDropout( theta, data, params )

[ W, b ] = stack2param(theta, params.decodeInfo);

numOrigImages = size(data.imgs, 2);
data.imgs = [data.imgs repmat(data.zeroimgs(:, 1), 1, params.numReplicate)];
data.categories = [data.categories repmat(data.zerocategories(:, 1), 1, params.numReplicate)];

numImages = size(data.imgs, 2);
imageSize = size(data.imgs, 1);

% Set 50% of the terms to zero
data.imgs(randi(size(data.imgs(:), 1), 1, numImages * imageSize / 2)) = 0;

% Feedforward
h = bsxfun(@plus, W{1} * data.imgs, b{1});
w = data.wordTable(:, data.categories);

cost = 0.5 / numImages * (sum(sum((h - w).^2)));

% Backprop
del = (h- w) / numImages;
Wgrad = del * data.imgs';
bgrad = sum(del, 2);

grad = [ Wgrad(:); bgrad(:) ];

if params.doEvaluate == true
    doPrint = true;
    mapDoEvaluate(data.imgs(:,1:numOrigImages), data.categories(:,1:numOrigImages), data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.validImgs, data.validCategories, data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.testImgs, data.testCategories, data.testOriginalCategoryNames, data.testCategoryNames, data.testWordTable, theta, params, doPrint);
end

end
