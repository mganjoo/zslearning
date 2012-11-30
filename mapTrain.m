addpath ../toolbox/;
addpath ../toolbox/minFunc/;

%% Model Parameters
fields = {{'wordDataset',         'turian.200'}; % type of embedding dataset to use ('turian.200')
          {'imageDataset',        'cifar10'};    % CIFAR dataset type
          {'batchFilePrefix',     'mini_batch'}; % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'maxPass',             150};     % maximum number of passes through training data
          {'maxIter',             5};      % maximum number of minFunc iterations on a batch
          {'fixRandom',           false};  % whether to fix the random number generator
          {'outputPath',          'savedParams'}; % the path to output files to
          {'saveEvery',           10};     % number of passes after which we need to do intermediate saves
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
[imgs, ~, ~] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, 1);
trainParams.imageColumnSize = size(imgs, 1); % the length of the column representation of a raw image

%% Load word representations
disp('Loading word representations');
load(['word_data/' trainParams.wordDataset '/wordTable.mat'], 'wordTable');

trainParams.inputSize = size(imgs, 1);
trainParams.outputSize = size(wordTable, 1);

%% First check the gradient of our minimizer
dataToUse.imgs = rand(2, 5);
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
debugTheta = mapInitParameters(debugParams);
[~, ~, ~, ~] = minFunc( @(p) mapTrainingCost(p, dataToUse, debugParams), debugTheta, debugOptions);

%% Load validation batch
disp('Loading validation batch');
[validImgs, ~, ~] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, numBatches+1);

%% Initialize actual weights
disp('Initializing parameters');
theta = mapInitParameters(trainParams);
dataToUse.wordTable = wordTable;

if not(exist(trainParams.outputPath, 'dir'))
    mkdir(trainParams.outputPath);
end

%% Begin batches of training
display(['Number of batches: ' num2str(numBatches)]);
statistics.costAfterBatch = zeros(1, numBatches * trainParams.maxPass);
for passj = 1:trainParams.maxPass
    for batchj = 1:numBatches        
        fprintf('----------------------------------------\n');
        fprintf('Pass %d, batch %d\n', passj, batchj);
        % optimize and gather statistics
        
        dataToUse.imgs = imgs;
        [theta, cost, ~, output] = minFunc( @(p) mapTrainingCost(p, dataToUse, trainParams), theta, options);
        statistics.costAfterBatch(1, (passj-1) * numBatches + batchj) = cost;
                        
        if batchj < numBatches
            nextBatch = batchj + 1;
        else
            nextBatch = 1;
        end
        
        if mod(passj, trainParams.saveEvery) == 0
            filename = sprintf('%s/params_pass_%d.mat', trainParams.outputPath, passj);
            save(filename, 'theta', 'trainParams');
        end

        [imgs, ~, ~] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, nextBatch);
    end
end

%% Save learned parameters
disp('Saving final learned parameters');
save(sprintf('%s/params_final.mat', trainParams.outputPath), 'theta', 'trainParams');
save(sprintf('%s/statistics.mat', trainParams.outputPath), 'statistics');
