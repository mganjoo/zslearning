function [ sortedOutlierIdxs ] = doOutlierDetection(method, XmapTrain, YmapTrain, XoutlierTrain, theta, trainParams, wordTable, topN, zeroCategories)

numCategories = size(wordTable, 2);
nonZeroCategories = setdiff(1:numCategories, zeroCategories);

mappedOutlierImages = mapDoMap(XoutlierTrain, theta, trainParams);
mappedTrainImages = mapDoMap(XmapTrain, theta, trainParams);

% Find top N neighbors for each category
if fullParams.topN ~= -1
    topNeighbors = zeros(length(nonZeroCategories), fullParams.topN);
    seenWordTable = wordTable(:, nonZeroCategories);
    for i = 1:length(nonZeroCategories)
        tDist = slmetric_pw(seenWordTable(:, i), mappedTrainImages(:, YmapTrain == nonZeroCategories(i)), 'eucdist');
        [~, topNeighbors(i, :) ] = sort(tDist);
    end
end

% Map back to original space if needed
if fullParams.outlierOriginalSpace
    mappedOutlierImages1 = XoutlierTrain;
    mappedTrainImages1 = XmapTrain;
    YmapTrain1 = YmapTrain;
    wordTable1 = zeros(size(mappedTrainImages1, 1), numCategories);
    for i = 1:length(nonZeroCategories)
        if fullParams.topN == -1
            wordTable1(:, i) = mean(mappedTrainImages1(:, YmapTrain1 == nonZeroCategories(i)), 2);
        else
            wordTable1(:, i) = mean(mappedTrainImages1(:, topNeighbors(i, 1:fullParams.topN)), 2);
        end
    end
else
    mappedOutlierImages1 = mappedOutlierImages;
    mappedTrainImages1 = mappedTrainImages;
    YmapTrain1 = YmapTrain;
    wordTable1 = wordTable;
end

if fullParams.topN ~= -1
    allIdxs = reshape(topNeighbors(:, 1:fullParams.topN), 1, []);
    mappedOutlierImages1 = mappedOutlierImages1(:, allIdxs);
    mappedTrainImages1 = mappedTrainImages1(:, allIdxs);
    YmapTrain1 = YmapTrain1(allIdxs);
end
    
if strcmp(method, 'gaussian')
    % Train Gaussian classifier
    disp('Training Gaussian classifier using Mixture of Gaussians');
    [mu, sigma, priors] = trainGaussianDiscriminant(mappedTrainImages1, YmapTrain1, numCategories, wordTable1);
    [~, sortedOutlierIdxs] = sort(predictGaussianDiscriminant(mappedOutlierImages1, mu, sigma, priors, zeroCategories));
elseif strcmp(method, 'gaussianPdf')
    % Train Gaussian classifier
    disp('Training Gaussian classifier using Mixture of Gaussians PDF');
    [mu, sigma, priors] = trainGaussianDiscriminant(mappedTrainImages1, YmapTrain1, numCategories, wordTable1);
    [~, sortedOutlierIdxs] = sort(predictGaussianDiscriminantMin(mappedOutlierImages1, mu, sigma, zeroCategories));
elseif strcmp(method, 'loop')
    disp('Training LoOP model');
    knn = 20;
    bestLambdas = [13, 10, 13, 12, 10, 10, 13, 10];
%     bestLambdas = randi(4, 1, length(nonZeroCategories)) + 8;
    [ nplofAll, pdistAll ] = trainOutlierPriors(mappedTrainImages1, YmapTrain1, nonZeroCategories, size(topNeighbors, 2), knn, bestLambdas);
    [~, sortedOutlierIdxs] = sort(calcOutlierPriors(mappedOutlierImages1, mappedTrainImages1, YmapTrain1, size(topNeighbors, 2), nonZeroCategories, bestLambdas, knn, nplofAll, pdistAll ), 'descend');
end

end

