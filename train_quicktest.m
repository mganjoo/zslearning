% Mini testing only

trainParams.wordDataset     = 'turian.200';
trainParams.imageDataset    = 'cifar96';
trainParams.batchFilePrefix = 'mini_batch';
trainParams.maxPass         = 3;
trainParams.maxIter         = 3;
trainParams.wReg            = 1E-3;
trainParams.iReg            = 1E-6;
trainParams.outputPath      = 'zsl-test';
trainParams.saveEvery       = 2;
train;
