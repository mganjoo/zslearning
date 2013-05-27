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

disp('Training mapping function');
% Train mapping function
trainParams.imageDataset = fullParams.dataset;
[theta, trainParams ] = fastTrain(XmapTrain, YmapTrain, trainParams, wordTable);
save(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');
% Get train accuracy
mapDoEvaluate(XmapTrain, YmapTrain, label_names, label_names, wordTable, theta, trainParams, true);

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
trainParamsSoftmax.sortedOutlierIdxs = sortedOutlierIdxs;
trainParamsSoftmax.nonZeroShotCategories = nonZeroCategories;
trainParamsSoftmax.allCategories = 1:numCategories;
[thetaSoftmax, trainParamsSoftmax] = combinedShotTrain(XoutlierTrain, YoutlierTrain, guessedZeroLabels, trainParamsSoftmax, label_names(nonZeroCategories));
save(sprintf('%s/thetaSoftmax.mat', outputPath), 'thetaSoftmax', 'trainParamSoftmax');

% Evaluate our trained softmax
softmaxDoEvaluate( testX, testY, label_names, thetaSoftmax, thetaParamsSoftmax, true );
