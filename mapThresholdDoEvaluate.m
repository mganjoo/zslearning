function [ guessedCategoriesDebug, results ] = mapThresholdDoEvaluate( images, categories, zeroCategoryTypes, categoryNames, maxDist, wordTable, theta, trainParams, doPrint )

addpath toolbox/pwmetric;

[ W, b ] = stack2param(theta, trainParams.decodeInfo);

numImages = size(images, 2);
numCategories = size(wordTable, 2);

% Remove zero-shot classes from word table
keep = arrayfun(@(x) ~ismember(x, zeroCategoryTypes), 1:length(categoryNames));

% Feedforward
mappedImages = bsxfun(@plus, 0.5 * W{1} * images, b{1});

% Find distances to word vectors
dist = slmetric_pw(wordTable, mappedImages, 'eucdist');
[ ~, guessedCategories ] = min(dist);

covered = bsxfun(@gt, maxDist, dist); % whether the new images are "covered" by existing categories
covered(zeroCategoryTypes, :) = true;

% check if classified class "covers" the image
% if not, then classify among zero shot only
unseenWordTable = wordTable(:, ~keep);
c = covered(:);
uncoveredImgIndices = ~c(guessedCategories + (0:numImages-1) * size(wordTable, 2));

if nnz(uncoveredImgIndices) > 0
    tDist = slmetric_pw(unseenWordTable, mappedImages(:, uncoveredImgIndices), 'eucdist');
    [~, tGuessedCategories ] = min(tDist);
    guessedCategories(uncoveredImgIndices) = zeroCategoryTypes(tGuessedCategories);
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
numUnseen = sum(arrayfun(@(x) nnz(categories == x), zeroCategoryTypes));
results.unseenAccuracy = sum(truePos(zeroCategoryTypes)) / numUnseen;
results.seenAccuracy = (sum(truePos) - sum(truePos(zeroCategoryTypes))) / (numImages - numUnseen);
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

