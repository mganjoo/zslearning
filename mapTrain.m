addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;

%% Model Parameters
fields = {{'wordDataset',         'acl'}; % type of embedding dataset to use ('turian.200', 'acl')
          {'imageDataset',        'cifar10'};    % CIFAR dataset type
          {'batchFilePrefix',     'mini_batch'}; % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'zeroFilePrefix',      'zeroshot_mini_batch'}; % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'maxPass',             1};     % maximum number of passes through training data
          {'maxIter',             400};      % maximum number of minFunc iterations on a batch
          {'fixRandom',           false};  % whether to fix the random number generator
          {'outputPath',          'savedParams'}; % the path to output files to
          {'lambda',              1E-3};  % regularization parameter
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

% minFunc options
options.Method = 'lbfgs';
options.display = 'on';
options.MaxIter = trainParams.maxIter;

% Additional options
batchFilePath   = ['image_data/batches/' trainParams.imageDataset];
files = dir([batchFilePath '/' trainParams.batchFilePrefix '*.mat']);
numBatches = length(files) - 1;
clear files;

%% Load first batch of training images
disp('Loading first batch of training images and initializing parameters');
[imgs, categories, categoryNames] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, 1);
[zeroimgs, ~, ~] = loadBatch(trainParams.zeroFilePrefix, trainParams.imageDataset, 1);
trainParams.imageColumnSize = size(imgs, 1); % the length of the column representation of a raw image

%% Load word representations
disp('Loading word representations');
t = load(['word_data/' trainParams.wordDataset '/' trainParams.imageDataset '/wordTable.mat']);
wordTable = zeros(size(t.wordTable, 1), length(categoryNames));
for i = 1:length(categoryNames)
    j = ismember(t.label_names, categoryNames{i}) == true;
    wordTable(:, i) = t.wordTable(:, j);
end
clear t;

trainParams.inputSize = size(imgs, 1);
trainParams.outputSize = size(wordTable, 1);

%% First check the gradient of our minimizer
dataToUse.imgs = rand(2, 5);
dataToUse.zeroimgs = rand(2, 3);
dataToUse.categories = randi(5, 1, 5);
dataToUse.wordTable = wordTable(1:2, 1:5);
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
debugParams.doEvaluate = false;
[ debugTheta, debugParams.decodeInfo ] = mapInitParameters(debugParams);
[~, ~, ~, ~] = minFunc( @(p) mapTrainingCost(p, dataToUse, debugParams), debugTheta, debugOptions);

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
dataToUse.categoryNames = categoryNames;
[theta, cost, ~, output] = minFunc( @(p) mapTrainingCost(p, dataToUse, trainParams ), theta, options);

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

%% Save learned parameters
disp('Saving final learned parameters');
save(sprintf('%s/params_final.mat', trainParams.outputPath), 'theta', 'trainParams');
