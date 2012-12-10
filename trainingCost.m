function [costTotal, gradTotal] = trainingCost( theta, data, params )

% Extract our weight and bias matrices from the stack.
[W, b] = stack2param(theta, params.decodeInfo);

hiddenSize    = size(W{1}, 1);
wordLen       = size(data.wordTable, 1);
numCategories = size(data.wordTable, 2);
numImages     = size(data.imgs, 2);

%% Feedforward

% Weight vectors for word and image components
W1_word = W{1}(:, 1:wordLen);
W1_image = W{1}(:, wordLen+1:end);

% Hidden layer
% Input corresponding to word components
z2words = W1_word * data.wordTable;
% Input corresponding to image components
z2image = bsxfun(@plus, W1_image * data.imgs, b{1});

[ t1,t2 ] = meshgrid(1:numCategories, 1:numImages);
a2 = params.f(z2words(:,t1(:)) + z2image(:,t2(:)));

% Output layer
% output is the set of final activations for all word-image combinations
output  = W{2} * a2;

% Extract columns corresponding to "good" combinations
pgood = [ data.wgood; data.imgs ];
a2good = a2(:, data.goodIndices);
outputGood = output(data.goodIndices);

% Build 2-D matrix of hinge losses (numImages * numCategories)
outputGrouped = reshape(output, numImages, [])';
hingeLossesGrouped = max(0, 1 + bsxfun(@plus, -outputGood, outputGrouped))';
hingeLossesGrouped(data.goodIndices) = 0;

% Cost per image
costPerImage = sum(hingeLossesGrouped, 2);

% If costs of all images have been optimized (nothing to change)
if costPerImage == 0
    costTotal = 0;
    gradTotal = zeros(size(theta));
    return;
end

% Regularization
reg = params.lambda * (sum(sum(W1_word.^2))) + params.lambda * (sum(sum(W1_image.^2))) ...
    + (params.lambda + params.lambda) * sum(sum(W{2}.^2));

% Total cost
costTotal = 1/numImages * sum(costPerImage) + 0.5 * reg;

%% Backpropagation

% Calculate derivative components
fpa2 = params.f_prime(a2);
fpa2good = fpa2(:, data.goodIndices);

% Define logical indexes for components of calculations we want to keep.
% We discard any components where the calculated hinge loss was zero.
keepIndices = reshape(hingeLossesGrouped > 0, 1, []);
a2(:, ~keepIndices) = 0;
fpa2(:, ~keepIndices) = 0;
totalkeep = sum(reshape(keepIndices', numImages, []), 2)';

% Calculate sums across blocks
sum_a2   = reshape(sum(reshape(a2, hiddenSize * numImages, []), 2), hiddenSize, []);
sum_fpa2 = reshape(sum(reshape(fpa2, hiddenSize * numImages, []), 2), hiddenSize, []);
sum_a2good = bsxfun(@times, totalkeep, a2good);
sum_fpa2good = bsxfun(@times, totalkeep, fpa2good);

% Gradient of W2
gradW2 = sum(-sum_a2good + sum_a2, 2)';

% Gradient of W1
t1 = reshape(sum(reshape(fpa2', numImages, [])), numCategories, [])' * data.wordTable';
t2 = sum_fpa2 * data.imgs';
gradW1 = bsxfun(@times, W{2}', -sum_fpa2good*pgood' + [t1 t2]);

% Gradient of b1
gradb1 = W{2}' .* sum(-sum_fpa2good + sum_fpa2, 2);

%% Update gradients
gradW1 = 1/numImages*gradW1 + [ params.lambda*W1_word params.lambda*W1_image ];
gradW2 = 1/numImages*gradW2 + (params.lambda+params.lambda)*W{2};
gradb1 = 1/numImages*gradb1;

gradTotal = [ gradW1(:); gradW2(:); gradb1(:) ];

if params.doEvaluate == true
    doPrint = true;
    mapDoEvaluate(data.imgs, data.categories, data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.validImgs, data.validCategories, data.categoryNames, data.categoryNames, data.wordTable, theta, params, doPrint);
    mapDoEvaluate(data.testImgs, data.testCategories, data.testOriginalCategoryNames, data.testCategoryNames, data.testWordTable, theta, params, doPrint);
end

end
