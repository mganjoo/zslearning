function [costTotal, gradTotal] = trainingCost(theta, data, params )

% Extract our weight and bias matrices from the stack.
[W, b] = stack2param(theta, params.decodeInfo);

wordVectorLength  = size(data.wordTable, 1);
numCategories     = size(data.wordTable, 2);
numImages         = size(data.imgs, 2);

%% Feedforward

% Weight vectors for word and image components
W1_word = W{1}(:, 1:wordVectorLength);
W1_image = W{1}(:, wordVectorLength+1:end);

% Hidden layer
% Input corresponding to word components
z2words = W1_word * data.wordTable;
% Input corresponding to image components
z2image = bsxfun(@plus, W1_image * data.imgs, b{1});

% [ [ z_w_i repeated m times for each image ] for all k categories ]
t1 = z2words(:, reshape(repmat(1:numCategories, numImages, 1), 1, []));
% [ z_im_1 .. z_im_m (all images) ] repeated k times for each category
t2 = repmat(z2image, 1, numCategories);
% a2 is the set of all word-image combinations (activated by f)
a2 = params.f(t1 + t2);

% Output layer
% output is the set of final activations for all word-image combinations
output  = W{2} * a2;

% Extract columns corresponding to "good" combinations
pgood = [ data.p1(:, data.goodIndices); data.imgs ];
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
reg = 0.5 * params.lambda * (sum(sum(W{1}.^2)) + sum(sum(W{2}.^2)));

% Total cost
costTotal = 1/numImages * sum(costPerImage) + reg;

%% Backpropagation

% Define logical indexes for components of calculations we want to keep.
% We discard any components where the calculated hinge loss was zero.
keepIndices = reshape(hingeLossesGrouped > 0, 1, []);

% Gradient of W2
temp = (repmat(-a2good, 1, numCategories) + a2);
gradW2 = sum(temp(:, keepIndices), 2)';

% Gradient of W1 & b1
deltagood = repmat(W{2}, numImages, 1)' .* params.f_prime(a2good);
deltabad  = repmat(W{2}, numCategories * numImages, 1)' .* params.f_prime(a2);

dmask = repmat(keepIndices, size(deltagood, 1), 1);
gradb1 = sum(dmask.*repmat(-deltagood, 1, numCategories) + dmask.*deltabad, 2);

% TODO: try to eliminate loop (can't because of many matrix outer products)
gradW1 = zeros(size(W{1}));
for j = 1:numCategories
    range  = (j-1)*numImages+1:j*numImages;
    pmask  = repmat(keepIndices(range)', 1, size(data.p1, 1) + size(data.imgs, 1));
    p      = [ data.p1(:, range); data.imgs ];
    gradW1 = gradW1 - deltagood*(pmask.*pgood') + deltabad(:, range)*(pmask.*p');
end

%% Update gradients
gradW1 = 1/numImages*gradW1 + params.lambda*W{1};
gradW2 = 1/numImages*gradW2 + params.lambda*W{2};
gradb1 = 1/numImages*gradb1;

gradTotal = [ gradW1(:); gradW2(:); gradb1(:) ];

end
