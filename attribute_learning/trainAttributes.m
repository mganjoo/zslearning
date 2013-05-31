function [thetas, fullTrainParams] = trainAttributes(X, Y, attributes, assignments, trainParams)

fields = {{'imageDataset',        'cifar10'};
          {'costFunction',        @softmaxCost}; % training cost function
          {'lambda',              1E-3};         % regularization parameter
          {'maxIter',             20};
};
   
for i = 1:length(fields)
    if exist('trainParams','var') && isfield(trainParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        trainParams.(fields{i}{1}) = fields{i}{2};
    end
end

trainParams.f = @tanh;             % function to use in the neural network activations
trainParams.f_prime = @tanh_prime; % derivative of f
trainParams.imageColumnSize = size(X, 1);

% Initialize actual weights
disp('Initializing parameters');
trainParams.inputSize = trainParams.imageColumnSize;
trainParams.outputSize = 2;

globalStart = tic;

numAttributes = length(attributes);
numCategories = size(assignments, 2);
thetas = cell(numCategories, 1);
fullTrainParams = cell(numCategories, 1);

if ~ismac && isunix && matlabpool('size') == 0
    numCores = feature('numCores');
    if numCores > 8
        numCores = 8;
    end
    matlabpool('open', numCores);
end

parfor i = 1:numAttributes
    fullTrainParams{i} = trainParams;
    [ thetas{i}, fullTrainParams{i}.decodeInfo ] = initializeParameters(fullTrainParams{i});
    fprintf('Training attribute %d: "%s"\n', i, attributes{i});
    assignmentsForY = normalizeAttributeValue(assignments(i, Y));
    dataToUse = struct('imgs', X, 'categories', assignmentsForY);
    thetas{i} = trainLBFGS(fullTrainParams{i}, dataToUse, thetas{i});
end

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);
