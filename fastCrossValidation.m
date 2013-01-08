trainParams.imageDataset = 'cifar96';
trainParams.dropoutFraction = 1;
trainParams.numReplicate = 0;

trainParams.trainFunction = @trainLBFGS;
trainParams.costFunction = @mapOneShotCostDropout;
train;

trainParams.trainFunction = @trainSGD;
trainParams.costFunction = @sgdOneShotCostDropout;
train;

trainParams.trainFunction = @trainLBFGS;
trainParams.costFunction = @cwTrainingCost;
train;