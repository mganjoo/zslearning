function [theta, trainParams] = zeroShotTrain(trainParams)

addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

%% Model Parameters
fields = {{'wordDataset',         'acl'};            % type of embedding dataset to use ('turian.200', 'acl')
          {'imageDataset',        'cifar10'};        % CIFAR dataset type
          {'lambda',              1E-3};   % regularization parameter
          {'costFunction',        @softmaxCost}; % training cost function
          {'trainFunction',       @trainLBFGS}; % training function to use
          {'hiddenSize',          100};
          {'maxIter',             400};    % maximum number of minFunc iterations on a batch      
          {'numRandom',           100};
          {'stddev',              1};
          
          % options
          {'batchFilePrefix',     'default_batch'};  % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'zeroFilePrefix',      'zeroshot_batch'}; % batch for zero shot images
          {'fixRandom',           false};  % whether to fix the random number generator
          {'enableGradientCheck', false};  % whether to enable gradient check
          {'preTrain',            true};   % whether to train on non-zero-shot first
          {'reloadData',          true};   % whether to reload data when this script is called (disable for batch jobs)
          
          % Old parameters, just keep for compatibility
          {'saveEvery',           5};      % number of passes after which we need to do intermediate saves
          {'oneShotMult',         1.0};    % multiplier for one-shot multiplier
          {'autoencMultStart',    0.01};   % starting value for autoenc mult
          {'sparsityParam',       0.035};  % desired average activation of the hidden units.
          {'beta',                5};      % weight of sparsity penalty term
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

% Initialize actual weights
zeroCategories = trainParams.zeroShotCategories;
t = load(['word_data/' trainParams.wordDataset '/' trainParams.imageDataset '/wordTable.mat']);
wordTable = t.wordTable;
trainParams.inputSize = size(wordTable, 1);
trainParams.outputSize = length(zeroCategories);
[ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);
numRandom = trainParams.numRandom;
X = zeros(size(wordTable, 1), numRandom * length(zeroCategories));
Y = zeros(1, numRandom * length(zeroCategories));
for i = 1:length(zeroCategories)
    X(:, (i-1)*numRandom+1) = wordTable(:, zeroCategories(i));
    for j = 1:numRandom-1
        X(:, (i-1)*numRandom+j+1) = normrnd(wordTable(:, zeroCategories(i)), trainParams.stddev);
    end
    Y(:, (i-1)*numRandom+1:i*numRandom) = i;
end

ind = randperm(length(Y));
X = X(:, ind);
Y = Y(ind);

globalStart = tic;
dataToUse.imgs = X;
dataToUse.categories = Y;

options.Method = 'lbfgs';
options.display = 'on';
options.MaxIter = trainParams.maxIter;
[theta, ~, ~, ~] = minFunc( @(p) softmaxCost(p, dataToUse, trainParams ), theta, options);

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

end
