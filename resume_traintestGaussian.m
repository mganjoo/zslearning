addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

fields = {{'dataset',        'cifar10'};
          {'wordset',        'acl'};
          {'resolution',     11};
          {'loadBestLambdas' true};
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('fullParams','var') && isfield(fullParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        fullParams.(fields{i}{1}) = fields{i}{2};
    end
end

dataset = fullParams.dataset;
wordset = fullParams.wordset;

if strcmp(dataset, 'cifar10')
    TOTAL_NUM_TRAIN = 50000;
    TOTAL_NUM_PER_CATEGORY = 5000;
    numCategories = 10;
    if isfield(fullParams,'zeroCategories')
        zeroCategories = fullParams.zeroCategories;
    else
        % 'frog', 'truck'
        zeroCategories = [ 4, 10 ];
    end
elseif strcmp(dataset, 'cifar96')
    TOTAL_NUM_TRAIN = 48000;
    TOTAL_NUM_PER_CATEGORY = 500;
    numCategories = 96;
    if isfield(fullParams,'zeroCategories')
        zeroCategories = fullParams.zeroCategories;
    else
        % 'boy', 'lion', 'orange', 'train', 'couch', 'house' 
        zeroCategories = [ 12, 42, 52, 87, 26, 36 ];
    end
else
    TOTAL_NUM_TRAIN = 53000;
    TOTAL_NUM_PER_CATEGORY = 500;
    numCategories = 106;
    if isfield(fullParams,'zeroCategories')
        zeroCategories = fullParams.zeroCategories;
    else
        % 'forest', 'lobster', 'boy', 'truck', 'orange', 'cat'
        zeroCategories = [ 33, 44, 12, 106, 52, 100 ];
    end
end

if not(exist('skipLoad','var')) || skipLoad == false
    disp('Loading data');
    load(['image_data/features/' dataset '/train.mat']);
    load(['image_data/features/' dataset '/test.mat']);
    load(['word_data/' wordset '/' dataset '/wordTable.mat']);
end

zeroList = label_names(zeroCategories);
zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];
outputPath = sprintf('gauss_%s_%s_%s', dataset, wordset, zeroStr);

nonZeroCategories = setdiff(1:numCategories, zeroCategories);
numTrainNonZeroShot = (numCategories - length(zeroCategories)) / numCategories * TOTAL_NUM_TRAIN;
numTrainPerCat = 0.95 * numTrainNonZeroShot / length(nonZeroCategories);
numValidatePerCat = numTrainPerCat * 0.05 / 0.95;

load(sprintf('%s/perm.mat', outputPath), 't', 'v');
X = trainX(:, t);
Y = trainY(t);
Xvalidate = trainX(:, v);
Yvalidate = trainY(v);

load(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');
load(sprintf('%s/thetaSeenSoftmax.mat', outputPath), 'thetaSeen', 'trainParamsSeen');
load(sprintf('%s/thetaUnseenSoftmax.mat', outputPath), 'thetaUnseen', 'trainParamsUnseen');

disp('Training Gaussian classifier using Mixture of Gaussians');
% Train Gaussian classifier
mapped = mapDoMap(X, theta, trainParams);
[mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminant(mapped, mu, sigma, priors, zeroCategories));

% Test
mappedTestImages = mapDoMap(testX, theta, trainParams);

resolution = fullParams.resolution;
gSeenAccuracies = zeros(1, resolution);
gUnseenAccuracies = zeros(1, resolution);
gAccuracies = zeros(1, resolution);
numPerIteration = numTrainNonZeroShot / (resolution-1);
logprobabilities = predictGaussianDiscriminant(mappedTestImages, mu, sigma, priors, zeroCategories);
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

disp('Training Gaussian classifier using PDF');
% Train Gaussian classifier
mapped = mapDoMap(X, theta, trainParams);
[mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminantMin(mapped, mu, sigma, zeroCategories));

% Test
mappedTestImages = mapDoMap(testX, theta, trainParams);

resolution = fullParams.resolution;
pdfSeenAccuracies = zeros(1, resolution);
pdfUnseenAccuracies = zeros(1, resolution);
pdfAccuracies = zeros(1, resolution);
numPerIteration = numTrainNonZeroShot / (resolution-1);
logprobabilities = predictGaussianDiscriminantMin(mappedTestImages, mu, sigma, zeroCategories);
cutoffs = [ arrayfun(@(x) sortedLogprobabilities((x-1)*numPerIteration+1), 1:resolution-1) sortedLogprobabilities(end) ];
for i = 1:resolution
    cutoff = cutoffs(i);
    % Test Gaussian classifier
    fprintf('With cutoff %f:\n', cutoff);
    results = mapGaussianThresholdDoEvaluate( testX, testY, zeroCategories, label_names, wordTable, ...
        theta, trainParams, thetaSeen, trainParamsSeen, thetaUnseen, trainParamsUnseen, logprobabilities, cutoff, true);

    pdfSeenAccuracies(i) = results.seenAccuracy;
    pdfUnseenAccuracies(i) = results.unseenAccuracy;
    pdfAccuracies(i) = results.accuracy;
end
pdfSeenAccuracies = fliplr(pdfSeenAccuracies);
pdfUnseenAccuracies = fliplr(pdfUnseenAccuracies);
pdfAccuracies = fliplr(pdfAccuracies);

disp('Training LoOP model');
resolution = fullParams.resolution - 1;
thresholds = 0:(1/resolution):1;
knn = 20;
if fullParams.loadBestLambdas
    load(sprintf('%s/bestLambdas.mat', outputPath), 'bestLambdas');
else
    lambdas = 1:13;
    loopSeenAccuracies = zeros(length(lambdas), length(thresholds));
    loopUnseenAccuracies = zeros(length(lambdas), length(thresholds));
    loopAccuracies = zeros(length(lambdas), length(thresholds));
    nonZeroCategoryIdPerm = randperm(length(nonZeroCategories));
    bestLambdas = repmat(lambdas(round(length(lambdas)/2)), 1, length(nonZeroCategories));
    mappedValidationImages = mapDoMap(Xvalidate, theta, trainParams);

    for k = 1:length(nonZeroCategories)
        changedCategory = nonZeroCategoryIdPerm(k);
        for i = 1:length(lambdas)
            tempLambdas = bestLambdas;
            tempLambdas(changedCategory) = lambdas(i);
            disp(tempLambdas);
            [ nplofAll, pdistAll ] = trainOutlierPriors(mapped, Y, nonZeroCategories, numTrainPerCat, knn, tempLambdas);
            probs = calcOutlierPriors( mappedValidationImages, mapped, Y, numTrainPerCat, nonZeroCategories, tempLambdas, knn, nplofAll, pdistAll );
            for t = 1:length(thresholds)
                fprintf('Threshold %f: ', thresholds(t));
                [~, results] = anomalyDoEvaluate(thetaSeen, ...
                    trainParamsSeen, thetaUnseen, trainParamsUnseen, probs, Xvalidate, mappedValidationImages, Yvalidate, ...
                    thresholds(t), zeroCategories, nonZeroCategories, wordTable, false);
                loopSeenAccuracies(i, t) = results.seenAccuracy;
                loopUnseenAccuracies(i, t) = results.unseenAccuracy;
                loopAccuracies(i, t) = results.accuracy;
                fprintf('seen accuracy: %f, unseen accuracy: %f\n', results.seenAccuracy, results.unseenAccuracy);
            end
        end
        [~, t] = max(sum(loopAccuracies,2));
        bestLambdas(changedCategory) = t;
    end
    disp('Best:');
    disp(bestLambdas);
    save(sprintf('%s/bestLambdas.mat', outputPath), 'bestLambdas');
end

% Do it again, with best lambdas
loopSeenAccuracies = zeros(1, length(thresholds));
loopUnseenAccuracies = zeros(1, length(thresholds));
loopAccuracies = zeros(1, length(thresholds));
[ nplofAll, pdistAll ] = trainOutlierPriors(mapped, Y, nonZeroCategories, numTrainPerCat, knn, bestLambdas);
probs = calcOutlierPriors( mappedTestImages, mapped, Y, numTrainPerCat, nonZeroCategories, bestLambdas, knn, nplofAll, pdistAll );
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

disp('Run Bayesian pipeline');
[~, bayesianResult] = mapBayesianDoEvaluate(thetaSeen, thetaUnseen, ...
    theta, trainParamsSeen, trainParamsUnseen, trainParams, mapped, Y, testX, ...
    testY, bestLambdas, knn, nplofAll, pdistAll, numTrainPerCat, zeroCategories, nonZeroCategories, label_names, true);

save(sprintf('%s/out_%s.mat', outputPath, zeroStr), 'gSeenAccuracies', 'gUnseenAccuracies', 'gAccuracies', ...
    'loopSeenAccuracies', 'loopUnseenAccuracies', 'loopAccuracies', 'pdfSeenAccuracies', 'pdfUnseenAccuracies', ...
    'pdfAccuracies', 'bayesianResult');
