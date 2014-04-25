addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

fields = {{'dataset',        'cifar10'};
          {'wordset',        'acl'};
          {'resolution',     11};
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

disp('Training seen softmax features');
mappedCategories = zeros(1, numCategories);
mappedCategories(nonZeroCategories) = 1:numCategories-length(zeroCategories);
trainParamsSeen.nonZeroShotCategories = nonZeroCategories;
[thetaSeen, trainParamsSeen] = nonZeroShotTrain(XmapTrain, mappedCategories(YmapTrain), trainParamsSeen, label_names(nonZeroCategories));
save(sprintf('%s/thetaSeenSoftmax.mat', outputPath), 'thetaSeen', 'trainParamsSeen');
% Get train accuracy
softmaxDoEvaluate( XmapTrain, YmapTrain, label_names, thetaSeen, trainParamsSeen, true );

disp('Training unseen softmax features');
trainParamsUnseen.zeroShotCategories = zeroCategories;
trainParamsUnseen.imageDataset = fullParams.dataset;
trainParamsUnseen.wordDataset = fullParams.wordset;
[thetaUnseen, trainParamsUnseen] = zeroShotTrain(trainParamsUnseen);
save(sprintf('%s/thetaUnseenSoftmax.mat', outputPath), 'thetaUnseen', 'trainParamsUnseen');

mapped = mapDoMap(XmapTrain, theta, trainParams);
mappedTestImages = mapDoMap(testX, theta, trainParams);

disp('Training Gaussian classifier using Mixture of Gaussians');
% Train Gaussian classifier
% [mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y, numCategories, wordTable);

pp = struct('outlierOriginalSpace', false, 'topN', 140);
[~, outlierParams] = doOutlierDetection('gaussian', XmapTrain, YmapTrain, XoutlierTrain, theta, trainParams, wordTable, pp, zeroCategories);

sortedLogprobabilities = sort(predictGaussianDiscriminant(mapped, outlierParams.mu, outlierParams.sigma, outlierParams.priors, zeroCategories));

% Test

resolution = fullParams.resolution;
gSeenAccuracies = zeros(1, resolution);
gUnseenAccuracies = zeros(1, resolution);
gAccuracies = zeros(1, resolution);
numPerIteration = floor(length(sortedLogprobabilities) / (resolution-1));
logprobabilities = predictGaussianDiscriminant(mappedTestImages, outlierParams.mu, outlierParams.sigma, outlierParams.priors, zeroCategories);
cutoffs = [ arrayfun(@(x) sortedLogprobabilities((x-1)*numPerIteration+1), 1:resolution-1) sortedLogprobabilities(end) ];
for i = 1:resolution
    cutoff = cutoffs(i);
    % Test Gaussian classifier
    fprintf('With cutoff %f:\n', cutoff);
    results = mapGaussianThresholdDoEvaluate( testX, testY, zeroCategories, label_names, wordTable, ...
        theta, trainParams, thetaSeen, trainParamsSeen, thetaUnseen, trainParamsUnseen, logprobabilities, cutoff, true);

    gSeenAccuracies(i) = results.seenAccuracy;
    gUnseenAccuracies(i) = results.unseenAccuracy;
    gAccuracies(i) = results.accuracy;
end
gSeenAccuracies = fliplr(gSeenAccuracies);
gUnseenAccuracies = fliplr(gUnseenAccuracies);
gAccuracies = fliplr(gAccuracies);

% disp('Training Gaussian classifier using PDF');
% % Train Gaussian classifier
% mapped = mapDoMap(X, theta, trainParams);
% [mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y, numCategories, wordTable);
% sortedLogprobabilities = sort(predictGaussianDiscriminantMin(mapped, mu, sigma, zeroCategories));
% 
% % Test
% mappedTestImages = mapDoMap(testX, theta, trainParams);
% 
% resolution = fullParams.resolution;
% pdfSeenAccuracies = zeros(1, resolution);
% pdfUnseenAccuracies = zeros(1, resolution);
% pdfAccuracies = zeros(1, resolution);
% numPerIteration = floor(length(sortedLogprobabilities) / (resolution-1));
% logprobabilities = predictGaussianDiscriminantMin(mappedTestImages, mu, sigma, zeroCategories);
% cutoffs = [ arrayfun(@(x) sortedLogprobabilities((x-1)*numPerIteration+1), 1:resolution-1) sortedLogprobabilities(end) ];
% for i = 1:resolution
%     cutoff = cutoffs(i);
%     % Test Gaussian classifier
%     fprintf('With cutoff %f:\n', cutoff);
%     results = mapGaussianThresholdDoEvaluate( testX, testY, zeroCategories, label_names, wordTable, ...
%         theta, trainParams, thetaSeen, trainParamsSeen, thetaUnseen, trainParamsUnseen, logprobabilities, cutoff, true);
% 
%     pdfSeenAccuracies(i) = results.seenAccuracy;
%     pdfUnseenAccuracies(i) = results.unseenAccuracy;
%     pdfAccuracies(i) = results.accuracy;
% end
% pdfSeenAccuracies = fliplr(pdfSeenAccuracies);
% pdfUnseenAccuracies = fliplr(pdfUnseenAccuracies);
% pdfAccuracies = fliplr(pdfAccuracies);

disp('Training LoOP model');
resolution = fullParams.resolution - 1;
thresholds = 0:(1/resolution):1;
% lambdas = 1:13;
% knn = 20;
% loopSeenAccuracies = zeros(length(lambdas), length(thresholds));
% loopUnseenAccuracies = zeros(length(lambdas), length(thresholds));
% loopAccuracies = zeros(length(lambdas), length(thresholds));
% nonZeroCategoryIdPerm = randperm(length(nonZeroCategories));
% bestLambdas = repmat(lambdas(round(length(lambdas)/2)), 1, length(nonZeroCategories));
% mappedValidationImages = mapDoMap(Xvalidate, theta, trainParams);

% for k = 1:length(nonZeroCategories)
%     changedCategory = nonZeroCategoryIdPerm(k);
%     for i = 1:length(lambdas)
%         tempLambdas = bestLambdas;
%         tempLambdas(changedCategory) = lambdas(i);
%         disp(tempLambdas);
%         [ nplofAll, pdistAll ] = trainOutlierPriors(mapped, Y, nonZeroCategories, numTrainPerCat, knn, tempLambdas);
%         probs = calcOutlierPriors( mappedValidationImages, mapped, Y, numTrainPerCat, nonZeroCategories, tempLambdas, knn, nplofAll, pdistAll );
%         for t = 1:length(thresholds)
%             fprintf('Threshold %f: ', thresholds(t));
%             [~, results] = anomalyDoEvaluate(thetaSeen, ...
%                 trainParamsSeen, thetaUnseen, trainParamsUnseen, probs, Xvalidate, mappedValidationImages, Yvalidate, ...
%                 thresholds(t), zeroCategories, nonZeroCategories, wordTable, false);
%             loopSeenAccuracies(i, t) = results.seenAccuracy;
%             loopUnseenAccuracies(i, t) = results.unseenAccuracy;
%             loopAccuracies(i, t) = results.accuracy;
%             fprintf('seen accuracy: %f, unseen accuracy: %f\n', results.seenAccuracy, results.unseenAccuracy);
%         end
%     end
%     [~, t] = max(sum(loopAccuracies,2));
%     bestLambdas(changedCategory) = t;
% end
% disp('Best:');
% disp(bestLambdas);

% Do it again, with best lambdas
loopSeenAccuracies = zeros(1, length(thresholds));
loopUnseenAccuracies = zeros(1, length(thresholds));
loopAccuracies = zeros(1, length(thresholds));
% [ nplofAll, pdistAll ] = trainOutlierPriors(mapped, Y, nonZeroCategories, numTrainPerCat, knn, bestLambdas);

pp = struct('outlierOriginalSpace', false, 'topN', 3000);
[~, outlierParams] = doOutlierDetection('loop', XmapTrain, YmapTrain, XoutlierTrain, theta, trainParams, wordTable, pp, zeroCategories);
probs = calcOutlierPriors( mappedTestImages, mapped, YmapTrain, outlierParams.numPerCat, nonZeroCategories, outlierParams.bestLambdas, outlierParams.knn, outlierParams.nplofAll, outlierParams.pdistAll );
for t = 1:length(thresholds)
    fprintf('Threshold %f: ', thresholds(t));
            [~, results] = anomalyDoEvaluate(thetaSeen, ...
                trainParamsSeen, thetaUnseen, trainParamsUnseen, probs, testX, mappedTestImages, testY, ...
                thresholds(t), zeroCategories, nonZeroCategories, wordTable, false);
    loopSeenAccuracies(t) = results.seenAccuracy;
    loopUnseenAccuracies(t) = results.unseenAccuracy;
    loopAccuracies(t) = results.accuracy;
    fprintf('accuracy: %f, seen accuracy: %f, unseen accuracy: %f\n', results.accuracy, results.seenAccuracy, results.unseenAccuracy);
end
% save(sprintf('%s/bestLambdas.mat', outputPath), 'bestLambdas');

disp('Run Bayesian pipeline');
[~, bayesianResult] = mapBayesianDoEvaluate(thetaSeen, thetaUnseen, ...
    theta, trainParamsSeen, trainParamsUnseen, trainParams, mapped, YmapTrain, testX, ...
    testY, outlierParams.bestLambdas, outlierParams.knn, outlierParams.nplofAll, outlierParams.pdistAll, ...
    outlierParams.numPerCat, zeroCategories, nonZeroCategories, label_names, true);

% Now run Bayesian pipeline where we recheck things classified as seen
mappedOutlierImages = mapDoMap(XoutlierTrain, theta, trainParams);
mappedTrainImages = mapDoMap(XmapTrain, theta, trainParams);
guessedZeroLabels = zeroCategories(softmaxPredict( mappedOutlierImages, thetaUnseen, trainParamsUnseen ));
pp = struct('outlierOriginalSpace', false, 'topN', 140);
[sortedOutlierIdxs, ~] = doOutlierDetection('gaussian', XmapTrain, YmapTrain, XoutlierTrain, theta, trainParams, wordTable, pp, zeroCategories);
trainParamsCombined.sortedOutlierIdxs = sortedOutlierIdxs;
trainParamsCombined.nonZeroShotCategories = nonZeroCategories;
trainParamsCombined.allCategories = 1:numCategories;
[thetaCombined, trainParamsCombined] = combinedShotTrain(XoutlierTrain, YoutlierTrain, guessedZeroLabels, trainParamsCombined);
save(sprintf('%s/thetaCombined.mat', outputPath), 'thetaCombined', 'trainParamsCombined');
combinedResult = softmaxDoEvaluate( testX, testY, label_names, thetaCombined, trainParamsCombined, true, zeroCategories );

[~, bayesianResultCombined] = mapBayesianDoEvaluateCombined(thetaCombined, thetaUnseen, ...
    theta, trainParamsCombined, trainParamsUnseen, trainParams, mapped, YmapTrain, testX, ...
    testY, outlierParams.bestLambdas, outlierParams.knn, outlierParams.nplofAll, outlierParams.pdistAll, ...
    outlierParams.numPerCat, zeroCategories, nonZeroCategories, label_names, true);


save(sprintf('%s/out_%s.mat', outputPath, zeroStr), 'gSeenAccuracies', 'gUnseenAccuracies', 'gAccuracies', ...
    'loopSeenAccuracies', 'loopUnseenAccuracies', 'bayesianResult', 'combinedResult', 'bayesianResultCombined');
