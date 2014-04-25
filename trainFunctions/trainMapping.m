function [theta, trainParams] = trainMapping(X, Y, trainParams, wordTable)

addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

%% Model Parameters
fields = {{'wordDataset',         'acl'};            % type of embedding dataset to use ('turian.200', 'acl')
          {'lambda',              1E-4};   % regularization parameter
          {'numReplicate',        0};     % one-shot replication
          {'dropoutFraction',     1};    % drop-out fraction
          {'costFunction',        @mapTrainingCostOneLayer}; % training cost function
          {'trainFunction',       @trainLBFGS}; % training function to use
          {'hiddenSize',          200};
          {'maxIter',             500};    % maximum number of minFunc iterations on a batch
          {'maxPass',             1};      % maximum number of passes through training data
          {'disableAutoencoder',  true};   % whether to disable autoencoder
          {'maxAutoencIter',      50};     % maximum number of minFunc iterations on a batch
          
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
trainParams.doEvaluate = false;
trainParams.testFilePrefix = 'zeroshot_test_batch';
trainParams.autoencMult = trainParams.autoencMultStart;

trainParams.imageColumnSize = size(X, 1);

trainParams.costFunction = @mapTrainingCost;

% Initialize actual weights
disp('Initializing parameters');
trainParams.inputSize = trainParams.imageColumnSize;
trainParams.outputSize = size(wordTable, 1);
[ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);

globalStart = tic;
dataToUse.imgs = X;
dataToUse.categories = Y;
dataToUse.wordTable = wordTable;

theta = trainParams.trainFunction(trainParams, dataToUse, theta);

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

end
