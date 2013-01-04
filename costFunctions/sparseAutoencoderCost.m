function [cost,grad] = sparseAutoencoderCost(theta, data, params)                                         

% Extract our weight and bias matrices from the stack.
[W, b] = stack2param(theta, params.decodeInfo);

allImgs = [ data.imgs data.zeroimgs ];
numImages = size(allImgs, 2);

% Perform feedforward passes on all data
a2 = params.f(bsxfun(@plus, W{1} * allImgs, b{1}));
h = bsxfun(@plus, W{2} * a2, b{2});

% Calculate overall cost function
rhoHat = (sum(a2, 2) / numImages + 1) / 2; % scale to [0,1] since we're using tanh
sparsity = params.beta * sum(params.sparsityParam * log(params.sparsityParam ./ rhoHat) ...
    + (1 - params.sparsityParam) * log((1 - params.sparsityParam) ./ (1 - rhoHat)));
reg = 0.5 * params.lambda * (sum(sum(W{1} .^ 2)) + sum(sum(W{2} .^ 2)));
cost = 0.5 / numImages * sum(sum((h - allImgs) .^ 2)) + reg + sparsity;

% Find error signal terms
del3 = -(allImgs - h);
sparsityMult = params.beta * (-(params.sparsityParam ./ rhoHat) + (1 - params.sparsityParam) ./ (1 - rhoHat));
del2 = bsxfun(@plus, W{2}' * del3, sparsityMult * 0.5) .* params.f_prime(a2);

% Calculate gradients
W2grad = del3 * a2' / numImages + params.lambda * W{2};
b2grad = sum(del3, 2) / numImages;
W1grad = del2 * allImgs' / numImages + params.lambda * W{1};
b1grad = sum(del2, 2) / numImages;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
