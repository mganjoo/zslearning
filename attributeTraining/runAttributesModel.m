% Train and evaluate using the Lampert et al. model
% You should run this file.

addpath ../costFunctions/;
addpath ../toolbox/;
addpath ../toolbox/minFunc/;

load('../image_data/features/cifar10/train.mat');
load('../image_data/features/cifar10/test.mat');
load('../word_data/acl/cifar10/wordTable.mat', 'label_names');
load('attribute_data.mat');

X = trainX(:, t1);
Y = trainY(:, t1);
Xvalid = trainX(:, v);
Yvalid = trainY(:, v);

trainParams = struct;
[thetas, fullTrainParams] = trainAttributes(X, Y, attributes, assignments, trainParams);

allCategories = 1:10;
zeroCategories = [4, 10];
nonZeroCategories = setdiff(allCategories, zeroCategories);

evaluateAttributes(testX, testY, thetas, fullTrainParams, ...
    assignments, zeroCategories, nonZeroCategories, label_names, true);
