% this one just treats everything within a cutoff as of the current
% category

function [ cutoffs ] = mapBayesianDoEvaluateCV2(thetaMapping, mapTrainParams, images, ...
    categories, wordVectors, cutoffStep, centerType, zeroCategoryTypes, nonZeroCategoryTypes)

addpath toolbox;
addpath toolbox/pwmetric;

mappedImages = mapDoMap(images, thetaMapping, mapTrainParams);

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
    bestF1 = 0;
    while true
        guessedCategories = zeros(size(categories));
        % treat everything outside the current cicle as of
        % other categories
        guessedCategories(dists < currentCutoff) = currentCategory;        
        truePos = and(guessedCategories == categories, guessedCategories == currentCategory);
%         results.accuracy = sum(truePos) / numImages;
        results.p = sum(truePos) / sum(guessedCategories == currentCategory);
        results.r = sum(truePos) / sum(categories == currentCategory);
        results.f1 = 2 * results.p * results.r / (results.p + results.r);
        if results.f1 < bestF1
            break
        end
        bestF1 = results.f1;
%         bestAccuracy = results.accuracy;
        currentCutoff = currentCutoff + cutoffStep;
    end
    disp(currentCutoff);
    cutoffs(currentCategory) = currentCutoff;
end

end

