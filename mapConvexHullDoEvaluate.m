function [ guessedCategoriesDebug, results, additionalWordTable ] = mapConvexHullDoEvaluate( images, categories, zeroCategoryTypes, categoryNames, wordTable, theta, trainParams, doPrint )

addpath toolbox/pwmetric;

[ W, b ] = stack2param(theta, trainParams.decodeInfo);

numImages = size(images, 2);
numCategories = size(wordTable, 2);

% First build additional categories
dist = slmetric_pw(wordTable, wordTable, 'eucdist');
numAdditionalCategories = trainParams.numAdditionalCategories;
additionalWordTable = zeros(size(wordTable, 1), (numAdditionalCategories) * length(zeroCategoryTypes));
k = 1;
[ ~, rankedIndices] = sort(dist);
for j = 1:length(zeroCategoryTypes)
    d = dist(:, zeroCategoryTypes(j)) / sum(dist(:, zeroCategoryTypes(j)));
    for i = 1:numAdditionalCategories
        if i == zeroCategoryTypes(j)
            continue;
        end
        interpol = trainParams.distanceMultiplier * d(i);    
        additionalWordTable(:, k) = interpol * wordTable(:, zeroCategoryTypes(j)) + (1-interpol) * wordTable(:, rankedIndices(i + 1, j));
        k = k + 1;
    end
end

% Feedforward
mappedImages = bsxfun(@plus, 0.5 * W{1} * images, b{1});

dist = slmetric_pw([wordTable additionalWordTable], mappedImages, 'eucdist');
[ ~, guessedCategories ] = min(dist);

% Map additional categories to original
for i = 1:length(zeroCategoryTypes)
    guessedCategories(and(guessedCategories >= numCategories + (i-1) * numAdditionalCategories + 1, guessedCategories <= numCategories + i * numAdditionalCategories)) = zeroCategoryTypes(i);
end

guessedCategoriesDebug = [ dist; guessedCategories ];

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(categories == actual);
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
    displayConfusionMatrix(confusion, categoryNames);
end

end

