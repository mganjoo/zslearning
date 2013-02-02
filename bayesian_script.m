addpath toolbox/;
addpath anomalyFunctions/;

numCategories = 10;
zeroCategories = [ 4, 10 ];
nonZeroCategories = setdiff(1:numCategories, zeroCategories);
knn = 20;

% Load images
load('image_data/batches/cifar10/validation_both.mat');

% Load trained mapped images
load('mappedTrainDataSmall.mat');

% Load other parameters
t = load('map-goodParams/params_final.mat');
thetaMapping = t.theta;
mapTrainParams = t.trainParams;

t = load('map-softmaxGoodParams/params_final.mat');
thetaSeenSoftmax = t.theta;
seenSmTrainParams = t.trainParams;

t = load('map-unseenSoftmaxGoodParams/params_final.mat');
thetaUnseenSoftmax = t.theta;
unseenSmTrainParams = t.trainParams;

% Load unseen word table
t = load('word_data/acl/cifar10/wordTable.mat');
load('image_data/images/cifar10/meta.mat');
wordTable = t.wordTable;
unseenWordTable = t.wordTable(:, zeroCategories);
clear t;

% Train and test anomaly detector
lambdas = 1:10;
for lambda = lambdas
    fprintf('Lambda: %d\n', lambda);
    [ nplof, pdist ] = trainOutlierPriors(trainX, knn, lambda);
    [~, results] = mapBayesianDoEvaluate(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, trainX, X, ...
    Y, lambda, knn, nplof, pdist, zeroCategories, nonZeroCategories, label_names, true);
end
