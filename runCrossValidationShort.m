% Default, with acl
trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
mapTrain;

trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
trainParams.numReplicate = 20;
mapTrain;

trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
trainParams.numReplicate = 25;
mapTrain;

trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
trainParams.dropoutFraction = 0.7;
mapTrain;

trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
trainParams.numReplicate = 20;
trainParams.dropoutFraction = 0.7;
mapTrain;

trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
trainParams.numReplicate = 25;
trainParams.dropoutFraction = 0.7;
mapTrain;

trainParams.maxIter = 10;     % maximum number of minFunc iterations on a batch
% Repeat everything with turian.200
trainParams.wordDataset = 'turian.200';
mapTrain;
