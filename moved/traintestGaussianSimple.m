% Choose dataset
fullParams.dataset = 'cifar10';

% Set any additional hyperparameters here
trainParams.lambda = 1E-4;
trainParams.hiddenSize = 200;
trainParams.maxIter = 500;

traintestGaussian;