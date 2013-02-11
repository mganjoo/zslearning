addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

fields = {{'dataset',        'cifar10'};
          {'wordset',        'acl'};
          {'resolution',     10};
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
trainFrac = 1;

if strcmp(dataset, 'cifar10')
    TOTAL_NUM_TRAIN = 50000;
    TOTAL_NUM_PER_CATEGORY = 5000;
    numCategories = 10;
    if isfield(fullParams,'zeroCategories')
        zeroCategories = fullParams.zeroCategories;
    else
        % 'cat', 'truck'
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
outputPath = sprintf('gauss_%s_%s', dataset, wordset);

if not(exist('skipLoad','var')) || skipLoad == false
    disp('Loading data');
    load(['image_data/features/' dataset '/train.mat']);
    load(['image_data/features/' dataset '/test.mat']);
    load(['word_data/' wordset '/' dataset '/wordTable.mat']);
end
    
if not(exist(outputPath, 'dir'))
    mkdir(outputPath);
end

disp('Zero categories:');
disp(zeroCategories);
nonZeroCategories = setdiff(1:numCategories, zeroCategories);

numTrain = (numCategories - length(zeroCategories)) / numCategories * TOTAL_NUM_TRAIN;
numTrainPerCat = numTrain / length(nonZeroCategories);
t = zeros(1, numTrain);
for i = 1:length(nonZeroCategories)
    [ ~, t((i-1)*numTrainPerCat+1:i*numTrainPerCat) ] = find(trainY == nonZeroCategories(i));
end

% permute
order = randperm(numTrain);
t = t(order);
X = trainX(:, t);
Y = trainY(:, t);
save(sprintf('%s/perm.mat', outputPath), 't');

disp('Training mapping function');
% Train mapping function
trainParams.imageDataset = fullParams.dataset;
[theta, trainParams ] = fastTrain(X, Y, trainParams, wordTable);
save(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');

disp('Training seen softmax features');
mappedCategories = zeros(1, numCategories);
mappedCategories(nonZeroCategories) = 1:numCategories-length(zeroCategories);
trainParamsSeen.nonZeroShotCategories = nonZeroCategories;
[thetaSeen, trainParamsSeen] = nonZeroShotTrain(X, mappedCategories(Y), trainParamsSeen);
save(sprintf('%s/thetaSeenSoftmax.mat', outputPath), 'thetaSeen', 'trainParamsSeen');

disp('Training unseen softmax features');
trainParamsUnseen.zeroShotCategories = nonZeroCategories;
[thetaUnseen, trainParamsUnseen] = zeroShotTrain(trainParamsUnseen);
save(sprintf('%s/thetaUnseenSoftmax.mat', outputPath), 'thetaUnseen', 'trainParamsUnseen');

disp('Training Gaussian classifier');
% Train Gaussian classifier
mapped = mapDoMap(X, theta, trainParams);
[mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminantMin(mapped, mu, sigma, priors, zeroCategories));

% Test
resolution = fullParams.resolution;
seenAccuracies = zeros(1, resolution);
unseenAccuracies = zeros(1, resolution);
accuracies = zeros(1, resolution);
numPerIteration = numTrain / resolution;
mappedTestImages = mapDoMap(testX, theta, trainParams);
logprobabilities = predictGaussianDiscriminantMin(mappedTestImages, mu, sigma, priors, zeroCategories);
for i = 1:resolution
    cutoff = sortedLogprobabilities((i-1)*numPerIteration+1);
    % Test Gaussian classifier
    fprintf('With cutoff %f:\n', cutoff);
    results = mapGaussianThresholdDoEvaluate( testX, testY, zeroCategories, label_names, wordTable, ...
        theta, trainParams, thetaSeen, trainParamsSeen, thetaUnseen, trainParamsUnseen, logprobabilities, cutoff, true);

    seenAccuracies(i) = results.seenAccuracy;
    unseenAccuracies(i) = results.unseenAccuracy;
    accuracies(i) = results.accuracy;
end
zeroList = label_names(zeroCategories);
zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];
save(sprintf('%s/out_%s.mat', outputPath, zeroStr), 'seenAccuracies', 'unseenAccuracies', 'accuracies');
