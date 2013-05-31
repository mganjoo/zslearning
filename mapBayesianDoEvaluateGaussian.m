function [ guessedCategories, results ] = mapBayesianDoEvaluateGaussian(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, images, ...
    categories, resolution, cutoffIdx, mu, sigma, priors, zeroCategoryTypes, nonZeroCategoryTypes, categoryNames, doPrint)

addpath toolbox;

numImages = size(images, 2);
numCategories = length(zeroCategoryTypes) + length(nonZeroCategoryTypes);
Ws = stack2param(thetaSeenSoftmax, seenSmTrainParams.decodeInfo);
Wu = stack2param(thetaUnseenSoftmax, unseenSmTrainParams.decodeInfo);

mappedImages = mapDoMap(images, thetaMapping, mapTrainParams);

% This is the seen label classifier
probSeen = exp(Ws{1}*images); % k by n matrix with all calcs needed
probSeen = bsxfun(@rdivide,probSeen,sum(probSeen));
probSeenFull = zeros(numCategories, numImages);
probSeenFull(nonZeroCategoryTypes, :) = probSeen;

% This is the unseen label classifier
probUnseen = exp(Wu{1}*mappedImages); % k by n matrix with all calcs needed
probUnseen = bsxfun(@rdivide,probUnseen,sum(probUnseen));
probUnseenFull = zeros(numCategories, numImages);
probUnseenFull(zeroCategoryTypes, :) = probUnseen;

logprobabilities = predictGaussianDiscriminant(mappedImages, mu, sigma, priors, zeroCategoryTypes);
sortedLogprobabilities = sort(logprobabilities);
numPerIteration = floor(length(sortedLogprobabilities) / (resolution-1));
cutoff = sortedLogprobabilities((cutoffIdx-1)*numPerIteration+1);

probs = zeros(size(categories));
probs(logprobabilities < cutoff) = 1;
finalProbs = bsxfun(@times, probSeenFull, 1 - probs) + bsxfun(@times, probUnseenFull, probs);
[~, guessedCategories ] = max(finalProbs);

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

