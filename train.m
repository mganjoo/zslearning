addpath toolbox/;
addpath toolbox/minFunc/;

%% Model Parameters
fields = {{'batchFilePrefix',     'default_batch'}; % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'maxPass',             40};     % maximum number of passes through training data
          {'maxIter',             5};      % maximum number of minFunc iterations on a batch
          {'hiddenSize',          100};    % number of units in hidden layer
          {'cReg',                1E-3};   % regularization parameter (weight decay)
          {'fixRandom',           false};  % whether to fix the random number generator
          {'outputPath',          'savedParams'}; % the path to output files to
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('trainParams','var') && isfield(trainParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        trainParams.(fields{i}{1}) = fields{i}{2};
    end
end

% Fix the random number generator if needed
if trainParams.fixRandom == true
    RandStream.setGlobalStream(RandStream('mcg16807','Seed', 0));
end

trainParams.f = @tanh;             % function to use in the neural network activations
trainParams.f_prime = @tanh_prime; % derivative of f
trainParams.saveEvery = 4;         % number of passes after which we need to do intermediate passes

% minFunc options
options.Method = 'lbfgs';
options.display = 'on';
options.MaxIter = trainParams.maxIter;

% Additional options
batchFilePath   = 'image_data/cifar-10-features';
files = dir([batchFilePath '/' trainParams.batchFilePrefix '*.mat']);
numBatches = length(files) - 1;
assert(numBatches >= 1, 'Must have at least two batch files (one for training, one for validation)');
clear files;

%% Load first batch of training images
disp('Loading first batch of training images and initializing parameters');
[imgs, categories, categoryNames] = loadTrainBatch(trainParams.batchFilePrefix, 1, batchFilePath);
numCategories = length(categoryNames);
trainParams.imageColumnSize = size(imgs, 1); % the length of the column representation of a raw image

%% Load word representations
disp('Loading word representations');
load('wordrep/wordreps_orig.mat', 'oWe');
load('wordrep/vocab.mat', 'vocab');
trainParams.embeddingSize = size(oWe, 1);
wordTable = zeros(trainParams.embeddingSize, length(categoryNames));
for categoryIndex = 1:length(categoryNames)
    icategoryWord = find(ismember(vocab, categoryNames(categoryIndex)) == true);
    wordTable(:, categoryIndex) = oWe(:, icategoryWord);
end

%% First check the gradient of our minimizer
debugImgs = rand(2, 5);
debugCategories = randi(numCategories, 1, 5);
debugWordTable = wordTable(1:2, 1:numCategories);
dataToUse = prepareData(debugImgs, debugCategories, debugWordTable);
debugOptions = struct;
debugOptions.Method = 'lbfgs';
debugOptions.display = 'off';
debugOptions.DerivativeCheck = 'on';
debugOptions.maxIter = 1;
debugParams = struct;
debugParams.inputSize = 4;
debugParams.hiddenSize = 5;
debugParams.f = trainParams.f;
debugParams.f_prime = trainParams.f_prime;
debugParams.cReg = 1E-3;
[debugTheta, debugParams.decodeInfo] = initializeParameters(debugParams);
[~, ~, ~, ~] = minFunc( @(p) trainingCost(p, dataToUse, debugParams), debugTheta, debugOptions);

%% Load validation batch
disp('Loading validation batch');
[validImgs, validCategories, validCategoryNames] = loadTrainBatch(trainParams.batchFilePrefix, numBatches+1, batchFilePath);

%% Initialize actual weights
disp('Initializing parameters');
[theta, trainParams.decodeInfo] = initializeParameters(trainParams);

if not(exist(trainParams.outputPath, 'dir'))
    mkdir(trainParams.outputPath);
end

%% Begin batches of training
display(['Number of batches: ' num2str(numBatches)]);
statistics.costAfterBatch = zeros(1, numBatches * trainParams.maxPass);
statistics.accuracies = zeros(1, trainParams.maxPass);
statistics.avgPrecisions = zeros(1, trainParams.maxPass);
statistics.avgRecalls = zeros(1, trainParams.maxPass);
for passj = 1:trainParams.maxPass
    for batchj = 1:numBatches
        dataToUse = prepareData(imgs, categories, wordTable);
        
        fprintf('----------------------------------------\n');
        fprintf('Pass %d, batch %d\n', passj, batchj);
        % optimize and gather statistics
        [theta, cost, ~, output] = minFunc( @(p) trainingCost(p, dataToUse, trainParams), theta, options);
        statistics.costAfterBatch(1, (passj-1) * trainParams.maxPass + batchj) = cost;
        
        % test on current training batch
        doEvaluate(dataToUse.imgs, dataToUse.categories, categoryNames, wordTable, theta, trainParams);
                
        if batchj < numBatches
            nextBatch = batchj + 1;
        else
            nextBatch = 1;
        end
        
        [imgs, categories, categoryNames] = loadTrainBatch(trainParams.batchFilePrefix, nextBatch, batchFilePath);
    end
    % test on validation batch
    fprintf('----------------------------------------\n');
    fprintf('Validation after pass %d\n', passj);
    [ ~, results ] = doEvaluate(validImgs, validCategories, categoryNames, wordTable, theta, trainParams);
    statistics.accuracies(passj) = results.accuracy;
    statistics.avgPrecisions(passj) = results.avgPrecision;
    statistics.avgRecalls(passj) = results.avgRecall;
    
    % intermediate saves
    if mod(passj, trainParams.saveEvery) == 0
        filename = sprintf('%s/params_pass_%d.mat', trainParams.outputPath, trainParams.batchFilePrefix, passj);
        save(filename, 'theta', 'trainParams');
    end
    
    if results.accuracy >= 0.7
        break;
    end
end

%% Save learned parameters
disp('Saving final learned parameters');
save(sprintf('%s/params_final.mat', trainParams.outputPath),'theta','trainParams');
save(sprintf('%s/statistics.mat', trainParams.outputPath), 'statistics');
