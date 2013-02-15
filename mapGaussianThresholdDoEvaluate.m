function [ results ] = mapGaussianThresholdDoEvaluate ( images, categories, zeroCategoryTypes, categoryNames, wordTable, thetaMapping, trainParams, thetaSoftmaxSeen, trainParamsSeen, thetaSoftmaxUnseen, trainParamsUnseen, logprobabilities, maxLogprobability, doPrint )

addpath toolbox/pwmetric;

numImages = size(images, 2);
numCategories = size(wordTable, 2);

nonzeroCategoryTypes = setdiff(1:length(categoryNames), zeroCategoryTypes);

guessedCategories = feedforwardDiscriminant(thetaMapping, thetaSoftmaxSeen, thetaSoftmaxUnseen, trainParams, trainParamsSeen, trainParamsUnseen, logprobabilities, images, maxLogprobability, zeroCategoryTypes, nonzeroCategoryTypes, wordTable);

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

