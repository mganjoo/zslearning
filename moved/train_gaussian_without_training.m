addpath toolbox/;
addpath anomalyFunctions/;

load 'gauss_cifar10_acl_cat_truck/perm.mat';
load 'image_data/features/cifar10/train.mat';

base_dir = 'gauss_cifar10_acl_cat_truck';

X = trainX(:, t1);
Y = trainY(t1);

numCategories = 10;
zeroCategories = [ 4, 10 ];
nonZeroCategories = setdiff(1:numCategories, zeroCategories);
knn = 20;

% Load images
validX = trainX(:, v);
validY = trainY(v);

% Load other parameters
t = load([base_dir '/theta.mat']);
thetaMapping = t.theta;
mapTrainParams = t.trainParams;

t = load([base_dir '/thetaSeenSoftmax.mat']);
thetaSeenSoftmax = t.thetaSeen;
seenSmTrainParams = t.trainParamsSeen;

t = load([base_dir '/thetaUnseenSoftmax.mat']);
thetaUnseenSoftmax = t.thetaUnseen;
unseenSmTrainParams = t.trainParamsUnseen;

% Load unseen word table
t = load('word_data/acl/cifar10/wordTable.mat');
load('image_data/images/cifar10/meta.mat');
wordTable = t.wordTable;
unseenWordTable = t.wordTable(:, zeroCategories);
clear t;

mapped = mapDoMap(X, thetaMapping, mapTrainParams);
[mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminant(mapped, mu, sigma, priors, zeroCategories));

cutoffs = mapBayesianDoEvaluateCV3(thetaSeenSoftmax, thetaUnseenSoftmax, thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, X, ...
    Y, wordTable, 0.05, 1, zeroCategories, nonZeroCategories);

[ guessedCategories, results ] = mapBayesianDoEvaluateGaussian2(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, validX, ...
    validY, cutoffs, zeroCategories, nonZeroCategories, label_names, wordTable, true);
