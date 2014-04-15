% this one just treats everything within a cutoff as of the current
% category

function [ cutoffs ] = mapBayesianDoEvaluateCV3(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, images, ...
    categories, wordVectors, cutoffStep, centerType, zeroCategoryTypes, nonZeroCategoryTypes)

addpath toolbox;
addpath toolbox/pwmetric;

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

cutoffs = zeros(length(nonZeroCategoryTypes) + length(zeroCategoryTypes));
for c_i = 1:length(nonZeroCategoryTypes)
    currentCategory = nonZeroCategoryTypes(c_i);
    disp(['Current category: ' num2str(currentCategory)]);
    if (centerType == 0) % Centroid
        centerVector = mean(mappedImages(:, categories == currentCategory), 2);
    elseif (centerType == 1) % Word vector
        centerVector = wordVectors(:, currentCategory);
    end
    dists = slmetric_pw(centerVector, mappedImages, 'eucdist');
    currentCutoff = 0;
    bestAccuracy = 0;
%     bestF1 = 0;
    while true
        % treat everything outside the current cicle as of
        % unseen
        guessedSeen = zeros(size(categories));
        guessedSeen(dists < currentCutoff) = 1;
        actualSeen = categories == currentCategory;
%         probs = zeros(size(categories));
%         probs(dists > currentCutoff) = 1;
%         finalProbs = bsxfun(@times, probSeenFull, 1 - probs) + bsxfun(@times, probUnseenFull, probs);
%         [~, guessedCategories ] = max(finalProbs);

        truePos = actualSeen == guessedSeen ;
%         truePos = and(guessedCategories == categories, guessedCategories == currentCategory);
        results.accuracy = sum(truePos) / numImages;
%         results.p = sum(truePos) / sum(guessedCategories == currentCategory);
%         results.r = sum(truePos) / sum(categories == currentCategory);
%         results.f1 = 2 * results.p * results.r / (results.p + results.r);
%         if results.f1 < bestF1
        disp(results.accuracy);
        if results.accuracy < bestAccuracy
            break
        else
    %         bestF1 = results.f1;
            bestAccuracy = results.accuracy;
            currentCutoff = currentCutoff + cutoffStep;
        end
    end
    disp(currentCutoff);
    cutoffs(currentCategory) = currentCutoff;
end

end

