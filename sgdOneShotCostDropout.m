function [cost, theta] = sgdOneShotCostDropout( theta, alpha, data, params )

[ W, b ] = stack2param(theta, params.decodeInfo);

numOrigImages = size(data.imgs, 2);
data.imgs = [data.imgs repmat(data.zeroimgs, 1, params.numReplicate)];
data.categories = [data.categories repmat(data.zerocategories, 1, params.numReplicate)];

numImages = size(data.imgs, 2);

% Set x% of the terms to zero
imgs = data.imgs;
data.imgs(rand(size(data.imgs))>params.dropoutFraction) = 0;

for i = 1:numImages
    % Feedforward
    h = bsxfun(@plus, W{1} * data.imgs(:, i), b{1});
    w = data.wordTable(:, data.categories(i));

    % Backprop
    del = (h - w);
    Wgrad = del * data.imgs(:, i)';
    bgrad = del;

    W{1} = W{1} - alpha * Wgrad;
    b{1} = b{1} - alpha * bgrad;
end

% Calculate cost after full pass
h = bsxfun(@plus, W{1} * imgs, b{1});
w = data.wordTable(:, data.categories);
cost = 0.5 * (sum(sum((h - w).^2)));

[theta, ~] = param2stack(W,b);

if params.doEvaluate == true
    doPrint = true;
    mapDoEvaluate(data.imgs(:,1:numOrigImages), data.categories(:,1:numOrigImages), data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.validImgs, data.validCategories, data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.testImgs, data.testCategories, data.testOriginalCategoryNames, data.testCategoryNames, data.testWordTable, theta, params, doPrint);
    [~, results ] = mapDoEvaluate(data.testImgs, data.testCategories, data.testOriginalCategoryNames, data.randCategoryNames, data.randWordTable, theta, params, false);
    fprintf('Accuracy when using 50 random categories: %.2f\n', results.accuracy);
end

end
