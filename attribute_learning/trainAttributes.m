function [thetas, trainParams] = trainAttributes(X, Y, attributes, assignments, trainParams)

fields = {{'imageDataset',        'cifar10'};
          {'costFunction',        @softmaxCost}; % training cost function
          {'trainFunction',       @trainLBFGS};  % training function to use
          {'lambda',              1E-3};         % regularization parameter
          {'maxIter',             100};
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
[ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);

globalStart = tic;
dataToUse.imgs = X;

numAttributes = length(attributes);
numCategories = size(assignments, 2);
thetas = cell(numCategories, 1);
for i = 1:numAttributes
    fprintf('Training attribute %d: "%s"\n', i, attributes{i});
    assignmentsForY = normalizeAttributeValue(assignments(i, Y));
    dataToUse.categories = assignmentsForY;
    thetas{i} = trainParams.trainFunction(trainParams, dataToUse, theta);
    [ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);
end

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);
