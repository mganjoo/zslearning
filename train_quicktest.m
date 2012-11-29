% Mini testing only

trainParams.wordDataset     = 'turian.200';
trainParams.batchFilePrefix = 'mini_batch_96';
trainParams.maxPass         = 3;
trainParams.maxIter         = 3;
trainParams.wReg            = 1E-3;
trainParams.iReg            = 1E-6;
trainParams.outputPath      = 'savedParams-test';
trainParams.saveEvery       = 2;
train;
