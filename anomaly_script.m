addpath toolbox/;
addpath anomalyFunctions/;

numCategories = 10;
zeroCategories = [ 4, 10 ];
knn = 20;
nonZeroCategories = setdiff(1:numCategories, zeroCategories);

% Load images
load('image_data/batches/cifar10/validation_both.mat');

% Load trained mapped images
load('mappedTrainDataSmall.mat');

% Load other parameters
t = load('map-goodParams/params_final.mat');
thetaUnseenMapping = t.theta;
trainParams = t.trainParams;
[ W, b ] = stack2param(thetaUnseenMapping, trainParams.decodeInfo);
mappedImages = bsxfun(@plus, 0.5 * W{1} * X, b{1});

t = load('map-softmaxGoodParams/params_final.mat');
thetaSeenSoftmax = t.theta;
smTrainParams = t.trainParams;

% Load unseen word table
t = load('word_data/acl/cifar10/wordTable.mat');
load('image_data/images/cifar10/meta.mat');
wordTable = t.wordTable;
unseenWordTable = t.wordTable(:, zeroCategories);
clear t;

% Train and test Gaussian anomaly detector
[mu, sigma, priors] = trainGaussianDiscriminant(trainX, trainY, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminant(trainX, mu, sigma, priors, zeroCategories));

% Test
numTrain = size(trainX, 2);
oseenAccuracies = numTrain / 600;
ounseenAccuracies = numTrain / 600;

for i = 1:numTrain/600
    cutoff = sortedLogprobabilities((i-1)*600+1);

    % Test Gaussian classifier
    results = mapGaussianThresholdDoEvaluate( X, Y, zeroCategories, label_names, wordTable, ...
        thetaUnseenMapping, thetaSeenSoftmax, trainParams, smTrainParams, cutoff, mu, sigma, priors, false);

    oseenAccuracies(i) = results.seenAccuracy;
    ounseenAccuracies(i) = results.unseenAccuracy;
end

% Train and test anomaly detector
thresholds = 0.05:0.05:0.5;
seenAccuracies = zeros(10, length(thresholds));
unseenAccuracies = zeros(10, length(thresholds));
lambdas = 4:12;
for lambda = lambdas
    fprintf('Lambda: %d\n', lambda);
    [ nplof, pdist ] = trainOutlierPriors(trainX, knn, lambda);
    probs = calcOutlierPriors( mappedImages, trainX, lambda, knn, nplof, pdist );
    for t = 1:length(thresholds)
        fprintf('Threshold %f: ', thresholds(t));
        [~, results] = anomalyDoEvaluate(thetaSeenSoftmax, ...
            smTrainParams, probs, unseenWordTable, X, mappedImages, Y, ...
            thresholds(t), zeroCategories, nonZeroCategories, false);

        seenAccuracies(lambda, t) = results.seenAccuracy;
        unseenAccuracies(lambda, t) = results.unseenAccuracy;
        fprintf('seen accuracy: %f, unseen accuracy: %f\n', results.seenAccuracy, results.unseenAccuracy);
    end
end

% plot(0:0.1:0.9, [seenAccuracies; unseenAccuracies]);
% hold on;
% plot(0:0.1:0.9, [oseenAccuracies; ounseenAccuracies], '--');
plot(0:0.1:0.9, [unseenAccuracies]);
leg = cell(1, length(lambdas));
for i = 1:length(lambdas)
    leg{i} = num2str(lambdas(i));
end
legend(leg);