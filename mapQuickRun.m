% Mini testing only

trainParams.wordDataset     = 'acl';
trainParams.imageDataset    = 'cifar10';
trainParams.batchFilePrefix = 'mini_batch';
trainParams.zeroFilePrefix  = 'zeroshot_mini_batch';
trainParams.costFunction    = @sgdOneShotCostDropout; % training cost function
trainParams.trainFunction   = @trainSGD; % training function to use
trainParams.maxPass         = 3;
trainParams.maxIter         = 3;
trainParams.saveEvery       = 2;
trainParams.outputPath      = 'map-test';
trainParams.maxAutoencIter  = 10;
mapTrain;
