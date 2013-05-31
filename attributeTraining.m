addpath costFunctions/;
addpath attribute_learning/;
addpath toolbox/;
addpath toolbox/minFunc/;

load('image_data/features/cifar10/train.mat');
load('image_data/features/cifar10/test.mat');

X = trainX(:, t1);
Y = trainY(:, t1);
Xvalid = trainX(:, v);
Yvalid = trainY(:, v);

load('attribute_learning/attribute_data.mat');
trainParams = struct;
[thetas, fullTrainParams] = trainAttributes(X, Y, attributes, assignments, trainParams);

load('word_data/acl/cifar10/wordTable.mat', 'label_names');

allCategories = 1:10;
zeroCategories = [4, 10];
nonZeroCategories = setdiff(allCategories, zeroCategories);

evaluateAttributes(testX, testY, thetas, fullTrainParams, ...
    assignments, zeroCategories, nonZeroCategories, label_names, true);