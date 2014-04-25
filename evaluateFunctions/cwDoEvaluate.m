function [ guessedCategoriesDebug, results ] = cwDoEvaluate( images, categories, originalCategoryNames, testCategoryNames, testWordTable, theta, trainParams, doPrint )

[Wt, bt] = stack2param(theta, trainParams.decodeInfo);
numImages = size(images, 2);
numCategories = size(testWordTable, 2);
wordVectorLength  = size(testWordTable, 1);

% Weight vectors for word and image components
W1_word = Wt{1}(:, 1:wordVectorLength);
W1_image = Wt{1}(:, wordVectorLength+1:end);

% Input corresponding to word components
z2words = W1_word * testWordTable;
% Input corresponding to image components
z2image = bsxfun(@plus, W1_image * images, bt{1});

% [ [ z_w_i repeated m times for each image ] for all k categories ]
t1 = z2words(:, reshape(repmat(1:numCategories, numImages, 1), 1, []));
% [ z_im_1 .. z_im_m (all images) ] repeated k times for each category
t2 = repmat(z2image, 1, numCategories);
% a2 is the set of all word-image combinations (activated by f)
a2 = trainParams.f(t1 + t2);

output  = Wt{2} * a2;
outputGrouped = reshape(output, numImages, [])';

[ ~, guessedCategories ] = max(outputGrouped);

% map categories from originalCategoryNames to testCategoryNames
mappedCategorySet = zeros(1, length(originalCategoryNames));
for i = 1:length(originalCategoryNames)
    mappedCategorySet(i) = find(strcmp(originalCategoryNames{i}, testCategoryNames));
end
mappedCategories = arrayfun(@(x) mappedCategorySet(x), categories);

guessedCategoriesDebug = [ outputGrouped; mappedCategories; guessedCategories ];

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(mappedCategories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t)));

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, testCategoryNames);
end

end

