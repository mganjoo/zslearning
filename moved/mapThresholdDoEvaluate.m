function [ results ] = mapThresholdDoEvaluate( images, categories, zeroCategoryTypes, categoryNames, wordTable, thetaMapping, thetaAnomaly, thetaSvm, trainParams, doPrint )

addpath toolbox/pwmetric;

[ W, b ] = stack2param(thetaMapping, trainParams.mappingDecodeInfo);

numImages = size(images, 2);
numCategories = size(wordTable, 2);

% Feedforward
mappedImages = bsxfun(@plus, 0.5 * W{1} * images, b{1});

% Remove zero-shot classes from word table
% Build unseen word table
% TODO: Make a random word table with 50 nouns
keep = arrayfun(@(x) ~ismember(x, zeroCategoryTypes), 1:length(categoryNames));
unseenWordTable = wordTable(:, ~keep);
nonzeroCategories = setdiff(1:length(categoryNames), zeroCategoryTypes);
guessedCategories = zeros(size(categories));

% Determine if seen or unseen
[Wanomaly, banomaly] = stack2param(thetaAnomaly, trainParams.anomalyDecodeInfo);
h = bsxfun(@plus, Wanomaly{1} * mappedImages, banomaly{1});
t = sum((h - images).^2);
seenIndices = t < trainParams.cutoff;
unseenIndices = ~seenIndices;

% If seen
[~, gind] = max(thetaSvm*images(:, seenIndices), [], 1);
guessedCategories(seenIndices) = nonzeroCategories(gind);

% If unseen
tDist = slmetric_pw(unseenWordTable, mappedImages(:, unseenIndices), 'eucdist');
[~, tGuessedCategories ] = min(tDist);
guessedCategories(unseenIndices) = zeroCategoryTypes(tGuessedCategories);

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
    disp(['Seen Accuracy: ' num2str(results.seenAccuracy)]);
    disp(['Unseen Accuracy: ' num2str(results.unseenAccuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, categoryNames);
end

end

