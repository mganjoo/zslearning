function [cost, grad] = mapTrainingCost( theta, data, params )

W = reshape(theta(1:params.inputSize*params.outputSize), params.outputSize, params.inputSize);
b = theta(params.inputSize*params.outputSize+1:params.inputSize*params.outputSize+params.outputSize);

numCategories = size(data.wordTable, 2);
numImages     = size(data.imgs, 2);

% Feedforward
z2 = bsxfun(@plus, W * data.imgs, b);
h = params.f(z2);

[ t1,t2 ] = meshgrid(1:numCategories, 1:numImages);
diff = h(:,t2(:)) - data.wordTable(:,t1(:));
cost = 0.5 / numImages / 1000 * sum(sum((diff) .^ 2));

sumdiff = reshape(sum(reshape(diff, params.outputSize * numImages, []), 2), params.outputSize, numImages);

% Backprop
del = (sumdiff .* params.f_prime(h)) / numImages / 1000;
Wgrad = del * data.imgs';
bgrad = sum(del, 2);

grad = [ Wgrad(:); bgrad(:) ];

end
