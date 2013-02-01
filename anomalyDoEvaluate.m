function [guessedCategories, results] = anomalyDoEvaluate(thetaSeenSoftmax, ...
    smTrainParams, priorProbs, unseenWordTable, images, mappedImages, categories, threshold, zeroCategoryTypes, nonZeroCategoryTypes, doPrint)

numImages = size(images, 2);
numCategories = length(zeroCategoryTypes) + length(nonZeroCategoryTypes);
Ws = stack2param(thetaSeenSoftmax, smTrainParams.decodeInfo);

% Determine seen/unseen based on hard threshold
unseenIndices = priorProbs >= threshold;
seenIndices = ~unseenIndices;

% This is the seen label classifier
pred = exp(Ws{1}*images(:, seenIndices)); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
[~, ind] = max(pred);
guessedCategories(seenIndices) = nonZeroCategoryTypes(ind);

% This is the unseen label classifier
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
results.confusion = confusion;

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Seen Accuracy: ' num2str(results.seenAccuracy)]);
    disp(['Unseen Accuracy: ' num2str(results.unseenAccuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, categoryNames);
end

end
