addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;

%% Model Parameters
fields = {{'wordDataset',         'acl'}; % type of embedding dataset to use ('turian.200', 'acl')
          {'imageDataset',        'cifar10'};    % CIFAR dataset type
          {'batchFilePrefix',     'default_batch'}; % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'zeroFilePrefix',      'zeroshot_batch'}; % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'maxIter',             200};     % maximum number of minFunc iterations on a batch
          {'maxPass',             1};     % maximum number of passes through training data
          {'maxAutoencIter',      50};     % maximum number of minFunc iterations on a batch
          {'disableAutoencoder',  true};  % whether to disable autoencoder
          {'fixRandom',           false};  % whether to fix the random number generator
          {'lambda',              1E-3};  % regularization parameter
          {'oneShotMult',         5.0};   % multiplier for one-shot multiplier
          {'costFunction',        @mapOneShotCostNoAutoenc}; % training cost function
          {'autoencMultStart',    0.01};   % starting value for autoenc mult
          {'sparsityParam',       0.035}; % desired average activation of the hidden units.
          {'beta',                5};     % weight of sparsity penalty term
          {'saveEvery',           5};     % number of passes after which we need to do intermediate saves
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('trainParams','var') && isfield(trainParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        trainParams.(fields{i}{1}) = fields{i}{2};
    end
end

if not(isfield(trainParams, 'outputPath'))
    trainParams.outputPath = sprintf('map-%s-iter_%d-pass_%d-ae_%d-aeiter_%d-reg_%.0e-1s_%.1f', func2str(trainParams.costFunction), trainParams.maxIter, trainParams.maxPass, trainParams.disableAutoencoder, trainParams.maxAutoencIter, trainParams.lambda, trainParams.oneShotMult);
end

fprintf('<BEGIN_EXPERIMENT %s>\n', trainParams.outputPath);
disp('Parameters:');
disp(trainParams);

% Fix the random number generator if needed
if trainParams.fixRandom == true
    RandStream.setGlobalStream(RandStream('mcg16807','Seed', 0));
end

trainParams.f = @tanh;             % function to use in the neural network activations
trainParams.f_prime = @tanh_prime; % derivative of f
trainParams.doEvaluate = true;
trainParams.testFilePrefix = 'zeroshot_test_batch';
trainParams.autoencMult = trainParams.autoencMultStart;

% minFunc options
options.Method = 'lbfgs';
options.display = 'on';

% Additional options
batchFilePath   = ['image_data/batches/' trainParams.imageDataset];
files = dir([batchFilePath '/' trainParams.batchFilePrefix '*.mat']);
numBatches = length(files) - 1;
clear files;

%% Load first batch of training images
disp('Loading first batch of training images and initializing parameters');
[imgs, categories, categoryNames] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, 1);
[zeroimgs, zerocategories, zeroCategoryNames] = loadBatch(trainParams.zeroFilePrefix, trainParams.imageDataset, 1);
t = randperm(size(zeroimgs, 2));
zeroimgs = zeroimgs(:, t);
zerocategories = zerocategories(:, t);

%% Load word representations
disp('Loading word representations');
t = load(['word_data/' trainParams.wordDataset '/' trainParams.imageDataset '/wordTable.mat']);
wordTable = zeros(size(t.wordTable, 1), length(categoryNames) + length(zeroCategoryNames));
for i = 1:length(categoryNames)
    j = ismember(t.label_names, categoryNames{i}) == true;
    wordTable(:, i) = t.wordTable(:, j);
end
for i = 1:length(zeroCategoryNames)
    j = ismember(t.label_names, zeroCategoryNames{i}) == true;
    wordTable(:, i + length(categoryNames)) = t.wordTable(:, j);
    zerocategories = zerocategories + length(categoryNames);
end
clear t;

trainParams.inputSize = size(imgs, 1);
trainParams.outputSize = size(wordTable, 1);

%% First check the gradient of our minimizer
dataToUse.imgs = rand(4, 10);
dataToUse.zeroimgs = rand(4, 4);
dataToUse.categories = randi(5, 1, 10);
dataToUse.zerocategories = ones(1, 4) + 5;
dataToUse.wordTable = wordTable(1:4, 1:6);
debugOptions = struct;
debugOptions.Method = 'lbfgs';
debugOptions.display = 'off';
debugOptions.DerivativeCheck = 'on';
debugOptions.maxIter = 1;
debugParams = struct;
debugParams.inputSize = size(dataToUse.imgs, 1);
debugParams.outputSize = size(dataToUse.wordTable, 1);
debugParams.f = trainParams.f;
debugParams.f_prime = trainParams.f_prime;
debugParams.lambda = trainParams.lambda;
debugParams.beta = trainParams.beta;
debugParams.sparsityParam = trainParams.sparsityParam;
debugParams.autoencMult = 1E-2;
debugParams.oneShotMult = 5.0;
debugParams.doEvaluate = false;
debugParams.costFunction = trainParams.costFunction;
[ debugTheta, debugParams.decodeInfo ] = mapInitParameters(debugParams);
if not(trainParams.disableAutoencoder)
    [~, ~, ~, ~] = minFunc( @(p) sparseAutoencoderCost(p, dataToUse, debugParams), debugTheta, debugOptions);
end
[~, ~, ~, ~] = minFunc( @(p) debugParams.costFunction(p, dataToUse, debugParams), debugTheta, debugOptions);

%% Load validation batch
disp('Loading validation batch');
[dataToUse.validImgs, dataToUse.validCategories, ~] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, numBatches+1);

%% Load test images
disp('Loading test images');
dataset = trainParams.imageDataset;
[dataToUse.testImgs, dataToUse.testCategories, dataToUse.testOriginalCategoryNames] = loadBatch(trainParams.testFilePrefix, trainParams.imageDataset);

if strcmp(dataset, 'cifar10') == true
    testCategoryNames = loadCategoryNames({ 'truck' }, dataset);
elseif strcmp(dataset, 'cifar96') == true
    testCategoryNames = loadCategoryNames({ 'lion', 'orange', 'camel' }, dataset);
else
    error('Not a valid dataset');
end
w = load(['word_data/' trainParams.wordDataset '/' dataset '/wordTable.mat']);
trainParams.embeddingSize = size(w.wordTable, 1);
dataToUse.testWordTable = zeros(trainParams.embeddingSize, length(testCategoryNames));
for categoryIndex = 1:length(testCategoryNames)
    icategoryWord = ismember(w.label_names, testCategoryNames(categoryIndex)) == true;
    dataToUse.testWordTable(:, categoryIndex) = w.wordTable(:, icategoryWord);
end
dataToUse.testCategoryNames = testCategoryNames;

%% Initialize actual weights
disp('Initializing parameters');
[ theta, trainParams.decodeInfo ] = mapInitParameters(trainParams);
dataToUse.wordTable = wordTable;

if not(exist(trainParams.outputPath, 'dir'))
    mkdir(trainParams.outputPath);
end

globalStart = tic;

dataToUse.imgs = imgs;
dataToUse.zeroimgs = zeroimgs;
dataToUse.categories = categories;
dataToUse.zerocategories = zerocategories;
dataToUse.categoryNames = categoryNames;

if not(trainParams.disableAutoencoder)
    options.MaxIter = trainParams.maxAutoencIter;
    [theta, cost, ~, output] = minFunc( @(p) sparseAutoencoderCost(p, dataToUse, trainParams ), theta, options);
    save(sprintf('%s/autoenc_params.mat', trainParams.outputPath), 'theta', 'trainParams');
end

options.MaxIter = trainParams.maxIter;
initMult = trainParams.autoencMult;
for i = 1:trainParams.maxPass
    [theta, cost, ~, output] = minFunc( @(p) trainParams.costFunction(p, dataToUse, trainParams ), theta, options);
    trainParams.autoencMult = trainParams.autoencMult + (1-initMult) / trainParams.maxPass;
end

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

%% Save learned parameters
disp('Saving final learned parameters');
disp('<END_EXPERIMENT>');
save(sprintf('%s/params_final.mat', trainParams.outputPath), 'theta', 'trainParams');
clear;
