function [cost, grad] = mapTrainingCost( theta, data, params )

W = reshape(theta(1:params.inputSize*params.outputSize), params.outputSize, params.inputSize);
b = theta(params.inputSize*params.outputSize+1:params.inputSize*params.outputSize+params.outputSize);

numImages     = size(data.imgs, 2);

% Feedforward
h = bsxfun(@plus, W * data.imgs, b);
w = data.wordTable(:, data.categories);

cost = 0.5 / numImages * sum(sum((h-w).^2));

% Backprop
del = (h-w) / numImages;
Wgrad = del * data.imgs';
bgrad = sum(del, 2);

grad = [ Wgrad(:); bgrad(:) ];

if params.doEvaluate == true
    doPrint = true;
    mapDoEvaluate(data.imgs, data.categories, data.categoryNames, data.categoryNames, data.wordTable, [W(:); b(:)], params, doPrint);
    mapDoEvaluate(data.validImgs, data.validCategories, data.categoryNames, data.categoryNames, data.wordTable, [W(:); b(:)], params, doPrint);
    mapDoEvaluate(data.testImgs, data.testCategories, data.testOriginalCategoryNames, data.testCategoryNames, data.testWordTable, [W(:); b(:)], params, doPrint);
end

end
