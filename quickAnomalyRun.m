% Mini testing only

trainParams.wordDataset     = 'acl';
trainParams.imageDataset    = 'cifar10';
trainParams.batchFilePrefix = 'mini_batch';
trainParams.maxIter         = 3;
trainParams.saveEvery       = 1;
trainParams.outputPath      = 'anomaly-test';
anomalyTrain;
