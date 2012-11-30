% Mini testing only

trainParams.wordDataset     = 'turian.200';
trainParams.imageDataset    = 'cifar10';
trainParams.batchFilePrefix = 'mini_batch';
trainParams.maxPass         = 3;
trainParams.maxIter         = 3;
trainParams.saveEvery       = 2;
trainParams.outputPath      = 'map-test';
mapTrain;
