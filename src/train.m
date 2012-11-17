addpath ../toolbox/;
addpath ../toolbox/minFunc/;

%% Model Parameters
fields = {{'embeddingSize',       50};     % dimension of the word embeddings
          {'layers',              100};    % number of units in each hidden layer (by default we have only 1 layer, so this is just a real number)
          {'batchSize',           8000};   % number of windows to use in each mini-batch during training
          {'numBatches',          4};      % number of batches used in training (note: (n+1)th batch will be used for validation)
          {'maxPass',             40};     % maximum number of passes through training data
          {'maxIter',             5};      % maximum number of minFunc iterations on a batch
          {'fixRandom',           false};  % whether to fix the random number generator
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('trainParams','var') && isfield(trainParams,fields{i}{1})
        disp(['Warning, we use the previously defined parameter ' fields{i}{1}])
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
trainParams.lambda = 1E-3;         % weight decay parameter
trainParams.saveEvery = 4;         % number of passes after which we need to do intermediate passes

% minFunc options
options.Method = 'lbfgs';
options.display = 'on';
options.MaxIter = trainParams.maxIter;

% Additional options
batchFilePrefix = 'features_batch'; % could be features_batch or mini_batch
batchFilePath   = '../image_data/cifar-10-features';
assert(trainParams.numBatches >= 1);
numBatches = trainParams.numBatches;

%% Load first batch of training images
disp('Loading first batch of training images and initializing parameters');
[imgs, categories, categoryNames] = loadCIFAR10TrainBatch(batchFilePrefix, 1, batchFilePath);
numCategories = length(categoryNames);
trainParams.imageColumnSize = size(imgs, 1); % the length of the column representation of a raw image

%% Load word representations
disp('Loading word representations');
load('../wordrep/wordreps_orig.mat', 'oWe');
load('../wordrep/vocab.mat', 'vocab');
wordVectorLength = size(oWe, 1);
wordTable = zeros(wordVectorLength, length(categoryNames));
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
debugParams.layers = 5;
debugParams.f = trainParams.f;
debugParams.f_prime = trainParams.f_prime;
debugParams.lambda = 1E-3;
[debugTheta, debugParams.decodeInfo] = initializeParameters(debugParams);
[~, ~, ~, ~] = minFunc( @(p) trainingCost(p, dataToUse, debugParams), debugTheta, debugOptions);

%% Load validation batch
disp('Loading validation batch');
[validImgs, validCategories, validCategoryNames] = loadCIFAR10TrainBatch(batchFilePrefix, numBatches+1, batchFilePath);

%% Initialize actual weights
disp('Initializing parameters');
[theta, trainParams.decodeInfo] = initializeParameters(trainParams);

if not(exist('../savedParams', 'dir'))
    mkdir('../savedParams');
end

%% Begin batches of training
display(['Number of batches: ' num2str(numBatches)]);
costAfterBatch = zeros(1, numBatches * trainParams.maxPass);
for passj = 1:trainParams.maxPass
    for batchj = 1:numBatches
        dataToUse = prepareData(imgs, categories, wordTable);
        
        fprintf('----------------------------------------\n');
        fprintf('Pass %d, batch %d\n', passj, batchj);
        % optimize and gather statistics
        [theta, cost, ~, output] = minFunc( @(p) trainingCost(p, dataToUse, trainParams), theta, options);
        costAfterBatch(1, (passj-1) * trainParams.maxPass + batchj) = cost;
        
        % test on current training batch
        doTest(dataToUse.imgs, dataToUse.categories, categoryNames, wordTable, theta, trainParams);
                
        if batchj < numBatches
            nextBatch = batchj + 1;
        else
            nextBatch = 1;
        end
        
        [imgs, categories, categoryNames] = loadCIFAR10TrainBatch(batchFilePrefix, nextBatch, batchFilePath);
    end
    % test on validation batch
    fprintf('----------------------------------------\n');
    fprintf('Validation after pass %d\n', passj);
    [ ~, accuracy ] = doTest(validImgs, validCategories, categoryNames, wordTable, theta, trainParams);
    
    % intermediate saves
    if mod(passj, trainParams.saveEvery) == 0
        filename = sprintf('../savedParams/params_pass_%d.mat', passj);
        save(filename, 'theta', 'trainParams');
    end
    
    if accuracy >= 0.7
        break;
    end
end

%% Save learned parameters
disp('Saving final learned parameters');
save('../savedParams/params_final.mat','theta','trainParams');
save('../savedParams/trainingStatistics.mat', 'costAfterBatch');
