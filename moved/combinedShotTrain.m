function [theta, trainParams] = combinedShotTrain(X, Y, guessedZeroLabels, trainParams)

addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

%% Model Parameters
fields = {{'wordDataset',         'acl'};            % type of embedding dataset to use ('turian.200', 'acl')
          {'imageDataset',        'cifar10'};        % CIFAR dataset type
          {'costFunction',        @softmaxCost}; % training cost function
          {'trainFunction',       @trainLBFGS}; % training function to use
          {'lambda',              1E-3};   % regularization parameter
          {'lambdaOld',           1};   % regularization parameter
          {'lambdaNew',           1E-3};   % regularization parameter
          {'numPretrainIter',     150};
          {'numSampleIter',       2};
          {'numTopOutliers',      15};
          {'numSampledNonZeroShot', 10};
          {'retrainCount',        10};
          {'outerRetrainCount',   10};
          
%           {'numReplicate',        0};     % one-shot replication
%           {'dropoutFraction',     1};    % drop-out fraction
%           {'hiddenSize',          100};
%           {'maxIter',             500};    % maximum number of minFunc iterations on a batch
%           {'maxPass',             1};      % maximum number of passes through training data
%           {'disableAutoencoder',  true};   % whether to disable autoencoder
%           {'maxAutoencIter',      50};     % maximum number of minFunc iterations on a batch

          % options
          {'fixRandom',           false};  % whether to fix the random number generator
          {'enableGradientCheck', false};  % whether to enable gradient check
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('trainParams','var') && isfield(trainParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        trainParams.(fields{i}{1}) = fields{i}{2};
    end
end

trainParams.f = @tanh;             % function to use in the neural network activations
trainParams.f_prime = @tanh_prime; % derivative of f
trainParams.doEvaluate = false;
trainParams.testFilePrefix = 'zeroshot_test_batch';

trainParams.imageColumnSize = size(X, 1);

% Initialize actual weights
disp('Initializing parameters');
nonZeroCategories = trainParams.nonZeroShotCategories;
allCategories = trainParams.allCategories;
zeroCategories = setdiff(allCategories, nonZeroCategories);
trainParams.inputSize = trainParams.imageColumnSize;
trainParams.outputSize = length(allCategories);
[ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);

globalStart = tic;

% Pretrain softmax
pretrainParams = trainParams;
pretrainParams.maxIter = trainParams.numPretrainIter;
nonZeroShotIdx = find(ismember(Y, nonZeroCategories));
dataToUse.imgs = X(:, nonZeroShotIdx);
dataToUse.categories = Y(nonZeroShotIdx);
dataToUse.nonZeroCategories = nonZeroCategories;
theta = trainParams.trainFunction(pretrainParams, dataToUse, theta);

% Find top N outliers
sampleTrainParams = trainParams;
sampleTrainParams.thetaOld = theta;
sampleTrainParams.outputsToChange = zeroCategories;
sampleTrainParams.costFunction = @softmaxCostMinimizeSeenChange;
for kk = 1:trainParams.outerRetrainCount
    fprintf('Outer iteration: %d\n', kk);
    for i = 1:trainParams.retrainCount
        fprintf('Iteration %d\n', i);
        pickedOutlierIdxs = trainParams.sortedOutlierIdxs((i-1)*trainParams.numTopOutliers+1:i*trainParams.numTopOutliers);
        outlierX = X(:, pickedOutlierIdxs);
        outlierY = guessedZeroLabels(pickedOutlierIdxs);
        % Subsample from remaining
        otherIdxs = intersect(nonZeroShotIdx, trainParams.sortedOutlierIdxs(trainParams.retrainCount*trainParams.numTopOutliers+1:end));
        sample = randi(length(otherIdxs), 1, trainParams.numSampledNonZeroShot);
        sampleX = X(:, sample);
        sampleY = Y(sample);

        XX = [outlierX sampleX];
        YY = [outlierY sampleY];
        perm = randperm(length(YY));
        XX = XX(:, perm);
        YY = YY(:, perm);

        sampleTrainParams.maxIter = trainParams.numSampleIter;
        dataToUse.imgs = XX;
        dataToUse.categories = YY;
        dataToUse.nonZeroCategories = nonZeroCategories;
        theta = trainParams.trainFunction(sampleTrainParams, dataToUse, theta);
    end
end

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

end
