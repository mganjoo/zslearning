function [cost,grad] = anomalyCost(theta, data, params)                                         

% Extract our weight and bias matrices from the stack.
[W, b] = stack2param(theta, params.decodeInfo);

numImages = size(data.imgs, 2);

% Perform feedforward passes on all data
a2 = data.mappedImgs;
h = bsxfun(@plus, W{1} * a2, b{1});

% Calculate overall cost function
reg = 0.5 * params.lambda * (sum(sum(W{1} .^ 2)) + sum(b{1} .^ 2));
cost = 0.5 / numImages * sum(sum((h - data.imgs) .^ 2)) + reg;

% Find error signal terms
del = -(data.imgs - h);

% Calculate gradients
Wgrad = del * a2' / numImages + params.lambda * W{1};
bgrad = sum(del, 2) / numImages;

grad = [Wgrad(:) ; bgrad(:)];
end
