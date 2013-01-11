addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

%% Model Parameters
fields = {{'wordDataset',         'acl'};            % type of embedding dataset to use ('turian.200', 'acl')
          {'imageDataset',        'cifar10'};        % CIFAR dataset type
          {'lambda',              1E-3};   % regularization parameter
          {'maxIter',             100};     % maximum number of minFunc iterations on a batch
          {'maxPass',             1};      % maximum number of passes through training data
          {'saveEvery',           20};     % save every
          
          % options
          {'batchFilePrefix',     'default_batch'};  % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'mappedImgFilename',   'mappedTrainX.mat'};
          {'zeroFilePrefix',      'zeroshot_batch'}; % batch for zero shot images
          {'fixRandom',           false};  % whether to fix the random number generator
          {'enableGradientCheck', true};  % whether to enable gradient check
          {'preTrain',            true};   % whether to train on non-zero-shot first
          {'costFunction',        @anomalyCost}; % training cost function
          {'trainFunction',       @trainLBFGS}; % training function to use
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

if not(isfield(trainParams, 'outputPath'))
    outputPath = sprintf('anomaly-%s-iter_%d-pass_%d-reg_%.0e-%s', ...
        trainParams.imageDataset, trainParams.maxIter, ...
        trainParams.maxPass, trainParams.lambda, datestr(now, 30));
else
    outputPath = trainParams.outputPath;
end

fprintf('<BEGIN_EXPERIMENT %s>\n', outputPath);
disp('Parameters:');
disp(trainParams);

% Fix the random number generator if needed
if trainParams.fixRandom == true
    RandStream.setGlobalStream(RandStream('mcg16807','Seed', 0));
end

trainParams.f = @tanh;             % function to use in the neural network activations
trainParams.f_prime = @tanh_prime; % derivative of f

[ imgs, ~, ~ ] = loadBatch(trainParams.batchFilePrefix, trainParams.imageDataset, 1);
batchFilePath   = ['image_data/batches/' trainParams.imageDataset];
load([batchFilePath '/' trainParams.mappedImgFilename]);

%% First check the gradient of our minimizer
if trainParams.enableGradientCheck
    ddataToUse.imgs = rand(4, 10);
    ddataToUse.mappedImgs = rand(2, 10);
    debugOptions.Method = 'lbfgs';
    debugOptions.display = 'off';
    debugOptions.DerivativeCheck = 'on';
    debugOptions.maxIter = 1;
    debugParams = trainParams;
    debugParams.inputSize = size(ddataToUse.mappedImgs, 1);
    debugParams.outputSize = size(ddataToUse.imgs, 1);
    [ debugTheta, debugParams.decodeInfo ] = initializeParameters(debugParams);
    [~, ~, ~, ~] = minFunc( @(p) debugParams.costFunction(p, ddataToUse, debugParams), debugTheta, debugOptions);
end 

% Initialize actual weights
disp('Initializing parameters');
trainParams.inputSize = size(mappedX, 1);
trainParams.outputSize = size(imgs, 1);
[ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);

if not(exist(outputPath, 'dir'))
    mkdir(outputPath);
end

globalStart = tic;
dataToUse.imgs = imgs;
dataToUse.mappedImgs = mappedX;
reps = trainParams.maxIter / trainParams.saveEvery;
trainParams.maxIter = trainParams.saveEvery;
for i = 1:reps
    theta = trainParams.trainFunction(trainParams, dataToUse, theta);
    disp('Saving intermediate parameters');
    save(sprintf('%s/params_%d.mat', outputPath, i), 'theta', 'trainParams');
end

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

%% Save learned parameters
disp('Saving final learned parameters');
save(sprintf('%s/params_final.mat', outputPath), 'theta', 'trainParams');
disp('<END_EXPERIMENT>');

