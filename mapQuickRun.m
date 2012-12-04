% Mini testing only

trainParams.wordDataset     = 'acl';
trainParams.imageDataset    = 'cifar10';
trainParams.batchFilePrefix = 'mini_batch';
trainParams.zeroFilePrefix  = 'zeroshot_mini_batch';
trainParams.maxPass         = 3;
trainParams.maxIter         = 3;
trainParams.saveEvery       = 2;
trainParams.outputPath      = 'map-test';
trainParams.maxAutoencIter  = 10;
mapTrain;
