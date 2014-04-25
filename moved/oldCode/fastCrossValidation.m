trainParams.imageDataset = 'cifar10';
trainParams.maxIter = 200;
trainParams.dropoutFraction = 1;
trainParams.numReplicate = 0;

trainParams.trainFunction = @trainLBFGS;
trainParams.costFunction = @mapOneShotCostDropout;
train;

trainParams.trainFunction = @trainSGD;
trainParams.costFunction = @sgdOneShotCostDropout;
train;

trainParams.imageDataset = 'cifar96';
trainParams.maxIter = 400;
trainParams.trainFunction = @trainLBFGS;
trainParams.costFunction = @cwTrainingCost;
train;

trainParams.dropoutFraction = 1;
trainParams.numReplicate = 0;
trainParams.trainFunction = @trainLBFGS;
trainParams.costFunction = @mapOneShotCostDropout;
train;

trainParams.trainFunction = @trainSGD;
trainParams.costFunction = @sgdOneShotCostDropout;
train;
