addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

fields = {{'dataset',       'cifar10'};
          {'wordset',       'acl'};
          {'outlierModel',  'gaussian'};
          {'resolution',    11};
          {'outlierOriginalSpace', true};
          {'unseenMethod', 'softmax'};
          {'loadOldParams', true};
          {'topN',          200}; % set to -1 to disable
          {'paramsPath', 'gauss_cifar10_acl_cat_truck_backup'};
          {'oracle',        false};
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

if ~fullParams.loadOldParams
    disp('Training mapping function');
    % Train mapping function
    trainParams.imageDataset = fullParams.dataset;
    [theta, trainParams ] = fastTrain(XmapTrain, YmapTrain, trainParams, wordTable);
    save(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');
    % Get train accuracy
    mapDoEvaluate(XmapTrain, YmapTrain, label_names, label_names, wordTable, theta, trainParams, true);
else
    load([fullParams.paramsPath '/theta.mat']);
end

% Now, train outlier model
mappedOutlierImages = mapDoMap(XoutlierTrain, theta, trainParams);
mappedTrainImages = mapDoMap(XmapTrain, theta, trainParams);

% Find the predictions for images assuming they're all zero-shot
if strcmp(fullParams.unseenMethod, 'map')
    unseenWordTable = wordTable(:, zeroCategories);
    tDist = slmetric_pw(unseenWordTable, mappedOutlierImages, 'eucdist');
    [~, tGuessedCategories ] = min(tDist);
    guessedZeroLabels = zeroCategories(tGuessedCategories);
elseif strcmp(fullParams.unseenMethod, 'softmax')
    load([fullParams.paramsPath '/thetaUnseenSoftmax.mat']);
    guessedZeroLabels = zeroCategories(softmaxPredict( mappedOutlierImages, thetaUnseen, trainParamsUnseen ));
end

sortedOutlierIdxs = doOutlierDetection(fullParams.outlierModel, XmapTrain, YmapTrain, XoutlierTrain, theta, trainParams, wordTable, fullParams.topN, zeroCategories);

if fullParams.oracle
    % Set up oracle prediction
    sortedOutlierIdxs = cell2mat(arrayfun(@(x) find(YoutlierTrain == x), [zeroCategories nonZeroCategories], 'UniformOutput', false));
    nonZeros = find(ismember(YoutlierTrain, nonZeroCategories));
    guessedZeroLabels = YoutlierTrain;
    guessedZeroLabels(nonZeros) = zeroCategories(randi(length(zeroCategories), 1, length(nonZeros)));
end

numNotOutliers = 1 - sum(ismember(YoutlierTrain(sortedOutlierIdxs(1:100)), zeroCategories)) / 100;
fprintf('%f fraction of the top 100 predicted outliers are not actually outliers.\n', numNotOutliers);

disp('Training softmax features');

% Cross validate
cvParams = {{'lambda',              [1E-2, 1E-3, 1E-4]};   % regularization parameter
            {'lambdaOld',           [1, 1E-1]};   % regularization parameter for seen weights during change
            {'lambdaNew',           [1E-3]};   % regularization parameter for unseen weights during change
            {'numPretrainIter',     [100, 150]};
            {'numSampleIter',       [2, 3]};
            {'numTopOutliers',      [15, 20, 40]};
            {'numSampledNonZeroShot', [2, 5, 10]};
            {'retrainCount',        [5, 10, 20]};
            {'outerRetrainCount',   [5, 10]};
            };
        
if isfield(fullParams, 'fixedCvParams')
    cvParams = fullParams.fixedCvParams;
end

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
    [thetaSoftmax, trainParamsSoftmax] = combinedShotTrain(XoutlierTrain, YoutlierTrain, guessedZeroLabels, trainParamsSoftmax);

    % Evaluate our trained softmax
    results = softmaxDoEvaluate( Xvalidate, Yvalidate, label_names, thetaSoftmax, trainParamsSoftmax, true, zeroCategories );
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
[thetaSoftmax, trainParamsSoftmax] = combinedShotTrain(XoutlierTrain, YoutlierTrain, guessedZeroLabels, trainParamsSoftmax );
save(sprintf('%s/thetaSoftmax.mat', outputPath), 'thetaSoftmax', 'trainParamsSoftmax');

fprintf('Best overall accuracy achieved with combination:\n');
disp(trainParamsSoftmax);
results = softmaxDoEvaluate( testX, testY, label_names, thetaSoftmax, trainParamsSoftmax, true, zeroCategories );

