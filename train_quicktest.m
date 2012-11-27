% Mini testing only

trainParams.wordDataset     = 'icml';
trainParams.batchFilePrefix = 'mini_batch';
trainParams.maxPass         = 2;
trainParams.maxIter         = 3;
trainParams.wReg            = 1E-3;
trainParams.iReg            = 1E-6;
trainParams.outputPath      = 'savedParams-test';
train;