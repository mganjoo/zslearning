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

end
