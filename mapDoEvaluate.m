function [ guessedCategoriesDebug, results ] = mapDoEvaluate( images, categories, originalCategoryNames, testCategoryNames, testWordTable, theta, trainParams, doPrint )

[ W, b ] = stack2param(theta, trainParams.decodeInfo);

numImages = size(images, 2);
numCategories = size(testWordTable, 2);

% Feedforward
mappedImages = trainParams.f(bsxfun(@plus, W{1} * images, b{1}));

dist = slmetric_pw(testWordTable, mappedImages, 'eucdist');
[ ~, guessedCategories ] = min(dist);

% map categories from originalCategoryNames to testCategoryNames
mappedCategorySet = zeros(1, length(originalCategoryNames));
for i = 1:length(originalCategoryNames)
    mappedCategorySet(i) = find(strcmp(originalCategoryNames{i}, testCategoryNames));
end
mappedCategories = arrayfun(@(x) mappedCategorySet(x), categories);

guessedCategoriesDebug = [ dist; mappedCategories; guessedCategories ];

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

