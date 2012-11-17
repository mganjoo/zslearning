function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, params, data)

% visibleSize: the number of input units
% hiddenSize: the number of hidden units
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units
% beta: weight of sparsity penalty term

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

m = size(data, 2);

% Perform feedforward passes on all data
z2 = bsxfun(@plus, W1 * data, b1);
a2 = params.f(z2);
z3 = bsxfun(@plus, W2 * a2, b2);
h  = z3; % since this is a linear decoder

% Calculate overall cost function
rhoHat = sum(a2, 2) / m;
sparsity = beta * sum(sparsityParam * log(sparsityParam ./ rhoHat) ...
+ (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - rhoHat)));
reg = 0.5 * lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));
cost = 0.5 / m * sum(sum((h - data) .^ 2)) + reg + sparsity;

% Find error signal terms
del3 = -(data - h);
sparsityMult = beta * (-(sparsityParam ./ rhoHat) + (1 - sparsityParam) ./ (1 - rhoHat));
del2 = ((W2' * del3) + repmat(sparsityMult, 1, m)) .* params.f_prime(a2);

% Calculate gradients
W2grad = del3 * a2' / m + lambda * W2;
b2grad = sum(del3, 2) / m;
W1grad = del2 * data' / m + lambda * W1;
b1grad = sum(del2, 2) / m;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
