addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

fields = {{'dataset',       'cifar10'};
          {'wordset',       'acl'};
          {'outlierModel',  'gaussian'};
          {'resolution',    11};
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('fullParams','var') && isfield(fullParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        fullParams.(fields{i}{1}) = fields{i}{2};
    end
end

loadDataRetrain;

% disp('Training mapping function');
% % Train mapping function
% trainParams.imageDataset = fullParams.dataset;
% [theta, trainParams ] = fastTrain(XmapTrain, YmapTrain, trainParams, wordTable);
% save(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');
% % Get train accuracy
% mapDoEvaluate(XmapTrain, YmapTrain, label_names, label_names, wordTable, theta, trainParams, true);

% We actually have saved theta and trainParams -- just use those for now
% (TODO)

% Now, train outlier model
mappedOutlierImages = mapDoMap(XoutlierTrain, theta, trainParams);
mappedTrainImages = mapDoMap(XmapTrain, theta, trainParams);

% Find the predictions for images assuming they're all zero-shot
unseenWordTable = wordTable(:, zeroCategories);
tDist = slmetric_pw(unseenWordTable, mappedOutlierImages, 'eucdist');
[~, tGuessedCategories ] = min(tDist);
guessedZeroLabels = zeroCategories(tGuessedCategories);

if strcmp(fullParams.outlierModel, 'gaussian')
    % Train Gaussian classifier
    disp('Training Gaussian classifier using Mixture of Gaussians');
    [mu, sigma, priors] = trainGaussianDiscriminant(mappedTrainImages, YmapTrain, numCategories, wordTable);
    [~, sortedOutlierIdxs] = sort(predictGaussianDiscriminant(mappedOutlierImages, mu, sigma, priors, zeroCategories));
elseif strcmp(fullParams.outlierModel, 'gaussianPdf')
    % Train Gaussian classifier
    disp('Training Gaussian classifier using Mixture of Gaussians PDF');
    [mu, sigma, priors] = trainGaussianDiscriminant(mappedTrainImages, YmapTrain, numCategories, wordTable);
    [~, sortedOutlierIdxs] = sort(predictGaussianDiscriminantMin(mappedOutlierImages, mu, sigma, priors, zeroCategories));
elseif strcmp(fullParams.outlierModel, 'loop')
    disp('Training LoOP model');
    knn = 20;
    bestLambdas = randi(4, 1, length(nonZeroCategories)) + 8;
    [ nplofAll, pdistAll ] = trainOutlierPriors(mappedTrainImages, YmapTrain, nonZeroCategories, numTrainMapPerCat, knn, bestLambdas);
    [~, sortedOutlierIdxs] = sort(calcOutlierPriors(mappedOutlierImages, mappedTrainImages, YmapTrain, numTrainMapPerCat, nonZeroCategories, bestLambdas, knn, nplofAll, pdistAll ), 'descend');
end

disp('Training softmax features');

% Cross validate
cvParams = {{'lambda',              [1E-3, 1E-4]};   % regularization parameter
            {'numPretrainIter',     [25, 50, 75]};
            {'numSampleIter',       [2, 3]};
            {'numTopOutliers',      [5, 10, 15]};
            {'numSampledNonZeroShot', [5, 10]};
            {'retrainCount',        [10, 20]};
            {'outerRetrainCount',   [5, 10]};
            };

combinations = buildCvParams(cvParams);
bestSeenAcc = 0;
bestUnseenAcc = 0;
bestOverallAcc = 0;
for kk = 1:length(combinations);
    trainParamsSoftmax = combinations(kk);
    disp(trainParamsSoftmax);
    trainParamsSoftmax.sortedOutlierIdxs = sortedOutlierIdxs;
    trainParamsSoftmax.nonZeroShotCategories = nonZeroCategories;
    trainParamsSoftmax.allCategories = 1:numCategories;
    [thetaSoftmax, trainParamsSoftmax] = combinedShotTrain(XoutlierTrain, YoutlierTrain, guessedZeroLabels, trainParamsSoftmax, label_names(nonZeroCategories));

    % Evaluate our trained softmax
    results = softmaxDoEvaluate( Xvalidate, Yvalidate, label_names, thetaSoftmax, trainParamsSoftmax, true );
    if results.seenAccuracy > bestSeenAcc
        bestSeenAccIdx = kk;
        bestSeenAcc = results.seenAccuracy;
    end
    if results.unseenAccuracy > bestUnseenAcc
        bestUnseenAccIdx = kk;
        bestUnseenAcc = results.unseenAccuracy;
    end
    if results.accuracy > bestOverallAcc
        bestAccIdx = kk;
        bestOverallAcc = results.accuracy;
    end
end

% Rerun on best overall accuracy index
trainParamsSoftmax = combinations(bestAccIdx);
trainParamsSoftmax.sortedOutlierIdxs = sortedOutlierIdxs;
trainParamsSoftmax.nonZeroShotCategories = nonZeroCategories;
trainParamsSoftmax.allCategories = 1:numCategories;
[thetaSoftmax, trainParamsSoftmax] = combinedShotTrain(XoutlierTrain, YoutlierTrain, guessedZeroLabels, trainParamsSoftmax, label_names(nonZeroCategories));
save(sprintf('%s/thetaSoftmax.mat', outputPath), 'thetaSoftmax', 'trainParamsSoftmax');

fprintf('Best overall accuracy achieved with combination:\n');
disp(trainParamsSoftmax);
results = softmaxDoEvaluate( testX, testY, label_names, thetaSoftmax, trainParamsSoftmax, true );

