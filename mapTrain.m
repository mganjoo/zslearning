addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;

%% Model Parameters
fields = {{'wordDataset',         'acl'};            % type of embedding dataset to use ('turian.200', 'acl')
          {'imageDataset',        'cifar10'};        % CIFAR dataset type
          {'lambda',              1E-3};   % regularization parameter
          {'numReplicate',        15};     % one-shot replication
          {'dropoutFraction',     0.5};    % drop-out fraction
          {'costFunction',        @sgdOneShotCostDropout}; % training cost function
          {'trainFunction',       @trainSGD}; % training function to use
          {'maxIter',             20};     % maximum number of minFunc iterations on a batch
          {'maxPass',             1};      % maximum number of passes through training data
          {'disableAutoencoder',  true};   % whether to disable autoencoder
          {'maxAutoencIter',      50};     % maximum number of minFunc iterations on a batch
          
          % options
          {'batchFilePrefix',     'default_batch'};  % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'zeroFilePrefix',      'zeroshot_batch'}; % batch for zero shot images
          {'fixRandom',           false};  % whether to fix the random number generator
          {'enableGradientCheck', false};  % whether to enable gradient check

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

if not(isfield(trainParams, 'outputPath'))
    trainParams.outputPath = sprintf('map-%s-%s-%s-iter_%d-pass_%d-noae_%d-aeiter_%d-reg_%.0e-1s_%d-dfrac_%.2f', ...
        func2str(trainParams.costFunction), trainParams.imageDataset, trainParams.wordDataset, trainParams.maxIter, ...
        trainParams.maxPass, trainParams.disableAutoencoder, trainParams.maxAutoencIter, trainParams.lambda, trainParams.numReplicate, ...
        trainParams.dropoutFraction);
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

if trainParams.enableGradientCheck
    %% First check the gradient of our minimizer
    dimgs = rand(4, 10);
    dcategories = randi(5, 1, 10);
    dwordTable = wordTable(1:4, 1:6);
    dataToUse = prepareData( dimgs, dcategories, dwordTable );
    dataToUse.zeroimgs = rand(4, 4);
    dataToUse.zerocategories = ones(1, 4) + 5;
    debugOptions.Method = 'lbfgs';
    debugOptions.display = 'off';
    debugOptions.DerivativeCheck = 'on';
    debugOptions.maxIter = 1;
    debugParams = trainParams;
    debugParams.autoencMult = 1E-2;
    debugParams.numReplicate = 3;
    debugParams.doEvaluate = false;
    debugParams.inputSize = size(dataToUse.imgs, 1);
    debugParams.outputSize = size(dataToUse.wordTable, 1);
    [ debugTheta, debugParams.decodeInfo ] = mapInitParameters(debugParams);
    if not(trainParams.disableAutoencoder)
        [~, ~, ~, ~] = minFunc( @(p) sparseAutoencoderCost(p, dataToUse, debugParams), debugTheta, debugOptions);
    end
    [~, ~, ~, ~] = minFunc( @(p) debugParams.costFunction(p, dataToUse, debugParams), debugTheta, debugOptions);
end 

% Prepare actual data
dataToUse = prepareData( imgs, categories, wordTable );
dataToUse.zeroimgs = zeroimgs;
dataToUse.zerocategories = zerocategories;
dataToUse.categoryNames = categoryNames;

%% Load validation batch
disp('Loading validation batch');
[dataToUse.validImgs, dataToUse.validCategories, ~] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, numBatches+1);

%% Load test images
disp('Loading test images');
dataset = trainParams.imageDataset;
[dataToUse.testImgs, dataToUse.testCategories, dataToUse.testOriginalCategoryNames] = loadBatch(trainParams.testFilePrefix, trainParams.imageDataset);

if strcmp(dataset, 'cifar10') == true
    testCategoryNames = loadCategoryNames({}, dataset);
elseif strcmp(dataset, 'cifar96') == true
    testCategoryNames = loadCategoryNames({ 'orange', 'camel' }, dataset);
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
trainParams.inputSize = size(imgs, 1);
trainParams.outputSize = size(wordTable, 1);
[ theta, trainParams.decodeInfo ] = mapInitParameters(trainParams);

if not(exist(trainParams.outputPath, 'dir'))
    mkdir(trainParams.outputPath);
end

globalStart = tic;

if not(trainParams.disableAutoencoder)
    options.MaxIter = trainParams.maxAutoencIter;
    [theta, ~, ~, ~] = minFunc( @(p) sparseAutoencoderCost(p, dataToUse, trainParams ), theta, options);
    save(sprintf('%s/autoenc_params.mat', trainParams.outputPath), 'theta', 'trainParams');
end

oldNumReplicate = trainParams.numReplicate;
trainParams.numReplicate = 0;
oldDropoutFraction = trainParams.dropoutFraction; 
trainParams.dropoutFraction = 1;
theta = trainParams.trainFunction(trainParams, dataToUse, theta);
disp('Saving training parameters');
save(sprintf('%s/train_params_final.mat', trainParams.outputPath), 'theta', 'trainParams');
trainParams.numReplicate = oldNumReplicate;
trainParams.dropoutFraction = oldDropoutFraction;
trainParams.maxIter = 5;
theta = trainParams.trainFunction(trainParams, dataToUse, theta);

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

%% Save learned parameters
disp('Saving final learned parameters');
disp('<END_EXPERIMENT>');
save(sprintf('%s/params_final.mat', trainParams.outputPath), 'theta', 'trainParams');
clear;
