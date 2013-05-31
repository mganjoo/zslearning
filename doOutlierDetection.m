function [ sortedOutlierIdxs, returnedParams ] = doOutlierDetection(method, XmapTrain, YmapTrain, XoutlierTrain, theta, trainParams, wordTable, params, zeroCategories)

numCategories = size(wordTable, 2);
nonZeroCategories = setdiff(1:numCategories, zeroCategories);

mappedOutlierImages = mapDoMap(XoutlierTrain, theta, trainParams);
mappedTrainImages = mapDoMap(XmapTrain, theta, trainParams);

% Find top N neighbors for each category
if params.topN ~= -1
    topNeighborsTrain = zeros(length(nonZeroCategories), params.topN);
    seenWordTable = wordTable(:, nonZeroCategories);
    for i = 1:length(nonZeroCategories)
        tDist = slmetric_pw(seenWordTable(:, i), mappedTrainImages(:, YmapTrain == nonZeroCategories(i)), 'eucdist');
        [~, tt ] = sort(tDist);
        topNeighborsTrain(i,:) = tt(1:params.topN);
    end
end

% Map back to original space if needed
if params.outlierOriginalSpace
    mappedOutlierImages1 = XoutlierTrain;
    mappedTrainImages1 = XmapTrain;
    YmapTrain1 = YmapTrain;
    wordTable1 = zeros(size(mappedTrainImages1, 1), numCategories);
    for i = 1:length(nonZeroCategories)
        if params.topN == -1
            wordTable1(:, i) = mean(mappedTrainImages1(:, YmapTrain1 == nonZeroCategories(i)), 2);
        else
            wordTable1(:, i) = mean(mappedTrainImages1(:, topNeighborsTrain(i, 1:params.topN)), 2);
        end
    end
else
    mappedOutlierImages1 = mappedOutlierImages;
    mappedTrainImages1 = mappedTrainImages;
    YmapTrain1 = YmapTrain;
    wordTable1 = wordTable;
end

if params.topN ~= -1
    allIdxs = unique(topNeighborsTrain(:, 1:params.topN));
    mappedTrainImages1 = mappedTrainImages1(:, allIdxs);
    YmapTrain1 = YmapTrain1(allIdxs);
else
    [~, t] = min(arrayfun(@(i) sum(YmapTrain1 == nonZeroCategories(i)), 1:length(nonZeroCategories)));
    topNeighborsTrain = find(YmapTrain1 == t);
end
    
if strcmp(method, 'gaussian')
    % Train Gaussian classifier
    disp('Training Gaussian classifier using Mixture of Gaussians');
    [mu, sigma, priors] = trainGaussianDiscriminant(mappedTrainImages1, YmapTrain1, numCategories, wordTable1);
    [~, sortedOutlierIdxs] = sort(predictGaussianDiscriminant(mappedOutlierImages1, mu, sigma, priors, zeroCategories));
    returnedParams.mu = mu;
    returnedParams.sigma = sigma;
    returnedParams.priors = priors;
elseif strcmp(method, 'gaussianPdf')
    % Train Gaussian classifier
    disp('Training Gaussian classifier using Mixture of Gaussians PDF');
    [mu, sigma, priors] = trainGaussianDiscriminant(mappedTrainImages1, YmapTrain1, numCategories, wordTable1);
    [~, sortedOutlierIdxs] = sort(predictGaussianDiscriminantMin(mappedOutlierImages1, mu, sigma, zeroCategories));
    returnedParams.mu = mu;
    returnedParams.sigma = sigma;
    returnedParams.priors = priors;
elseif strcmp(method, 'loop')
    disp('Training LoOP model');
    knn = 20;
    bestLambdas = [13, 10, 13, 12, 10, 10, 13, 10];
%     bestLambdas = randi(4, 1, length(nonZeroCategories)) + 8;
    [ nplofAll, pdistAll ] = trainOutlierPriors(mappedTrainImages1, YmapTrain1, nonZeroCategories, size(topNeighborsTrain, 2), knn, bestLambdas);
    [~, sortedOutlierIdxs] = sort(calcOutlierPriors(mappedOutlierImages1, mappedTrainImages1, YmapTrain1, size(topNeighborsTrain, 2), nonZeroCategories, bestLambdas, knn, nplofAll, pdistAll ), 'descend');
    returnedParams.nplofAll = nplofAll;
    returnedParams.pdistAll = pdistAll;
    returnedParams.knn = 20;
    returnedParams.bestLambdas = bestLambdas;
    returnedParams.numPerCat = size(topNeighborsTrain, 2);
end

end

