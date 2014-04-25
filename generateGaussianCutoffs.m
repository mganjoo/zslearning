function [ cutoffs ] = generateGaussianCutoffs(thetaSeenSoftmax, thetaUnseenSoftmax, ...
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
    % What vector do we treat as the center of the cluster?
    if (centerType == 0) % Centroid
        centerVector = mean(mappedImages(:, categories == currentCategory), 2);
    elseif (centerType == 1) % Word vector
        centerVector = wordVectors(:, currentCategory);
    end
    dists = slmetric_pw(centerVector, mappedImages, 'eucdist');
    currentCutoff = 0;
    bestAccuracy = 0;
    while true
        % treat everything outside the current cicle as of unseen
        guessedSeen = zeros(size(categories));
        guessedSeen(dists < currentCutoff) = 1;
        actualSeen = categories == currentCategory;

        truePos = actualSeen == guessedSeen ;
        results.accuracy = sum(truePos) / numImages;
        disp(results.accuracy);
        if results.accuracy < bestAccuracy
            break
        else
            bestAccuracy = results.accuracy;
            currentCutoff = currentCutoff + cutoffStep;
        end
    end
    disp(currentCutoff);
    cutoffs(currentCategory) = currentCutoff;
end

end

