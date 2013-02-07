function [cost, grad] = mapTrainingCostOneLayer( theta, data, params )

[ W, b ] = stack2param(theta, params.decodeInfo);

%numOrigImages = size(data.imgs, 2);
%data.imgs = [data.imgs repmat(data.zeroimgs, 1, params.numReplicate)];
%data.categories = [data.categories repmat(data.zerocategories, 1, params.numReplicate)];

numImages = size(data.imgs, 2);

% Set x% of the terms to zero
%data.imgs(rand(size(data.imgs))>params.dropoutFraction) = 0;

% Feedforward
h = bsxfun(@plus, W{1} * data.imgs, b{1});
w = data.wordTable(:, data.categories);

reg = 0.5 * params.lambda * (sum(sum(W{1} .^ 2)) + sum(b{1} .^ 2));
cost = 0.5 / numImages * (sum(sum((h - w).^2))) + reg;

% Backprop
del = (h- w) / numImages;
Wgrad = del * data.imgs' + params.lambda * W{1};
bgrad = sum(del, 2) + params.lambda * b{1};

grad = [ Wgrad(:); bgrad(:) ];

end
