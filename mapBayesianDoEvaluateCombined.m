function [ guessedCategories, results ] = mapBayesianDoEvaluateCombined(thetaCombined, thetaUnseenSoftmax, ...
    thetaMapping, combinedSmTrainParams, unseenSmTrainParams, mapTrainParams, trainX, trainY, images, ...
    categories, lambdas, knn, nplofAll, pdistAll, numPerCategory, zeroCategoryTypes, nonZeroCategoryTypes, categoryNames, doPrint)

addpath toolbox;

numImages = size(images, 2);
numCategories = length(zeroCategoryTypes) + length(nonZeroCategoryTypes);
Wc = stack2param(thetaCombined, combinedSmTrainParams.decodeInfo);
Wu = stack2param(thetaUnseenSoftmax, unseenSmTrainParams.decodeInfo);

mappedImages = mapDoMap(images, thetaMapping, mapTrainParams);

priors = calcOutlierPriors( mappedImages, trainX, trainY, numPerCategory, nonZeroCategoryTypes, lambdas, knn, nplofAll, pdistAll );

% This is the seen label classifier
probSeen = exp(Wc{1}*images); % k by n matrix with all calcs needed
probSeenFull = bsxfun(@rdivide,probSeen,sum(probSeen));

% This is the unseen label classifier
probUnseen = exp(Wu{1}*mappedImages); % k by n matrix with all calcs needed
probUnseen = bsxfun(@rdivide,probUnseen,sum(probUnseen));
probUnseenFull = zeros(numCategories, numImages);
probUnseenFull(zeroCategoryTypes, :) = probUnseen;

finalProbs = bsxfun(@times, probSeenFull, 1 - priors) + bsxfun(@times, probUnseenFull, priors);
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

