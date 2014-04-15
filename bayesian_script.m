addpath toolbox/;
addpath anomalyFunctions/;

numCategories = 10;
zeroCategories = [ 4, 10 ];
nonZeroCategories = setdiff(1:numCategories, zeroCategories);
knn = 20;

% Load images
load('image_data/batches/cifar10/validation_both.mat');

% Load trained mapped images
load('mappedTrainData.mat');
numImages = length(trainY);
numPerCategory = 3950;

% Load other parameters
t = load('parameter_archive/map-goodParams/params_final.mat');
thetaMapping = t.theta;
mapTrainParams = t.trainParams;

t = load('parameter_archive/map-softmaxGoodParams/params_final.mat');
thetaSeenSoftmax = t.theta;
seenSmTrainParams = t.trainParams;

t = load('parameter_archive/map-unseenSoftmaxGoodParams/params_final.mat');
thetaUnseenSoftmax = t.theta;
unseenSmTrainParams = t.trainParams;

% Load unseen word table
t = load('word_data/acl/cifar10/wordTable.mat');
load('image_data/images/cifar10/meta.mat');
wordTable = t.wordTable;
unseenWordTable = t.wordTable(:, zeroCategories);
clear t;

% Train and test anomaly detector
lambdas = 1:3;
for lambda = lambdas
    fprintf('Lambda: %d\n', lambda);
    [ nplofAll, pdistAll ] = trainOutlierPriors(trainX, trainY, nonZeroCategories, numPerCategory, knn, lambda);
    [~, results] = mapBayesianDoEvaluate(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, trainX, trainY, X, ...
    Y, lambda, knn, nplofAll, pdistAll, numPerCategory, zeroCategories, nonZeroCategories, label_names, true);
end
