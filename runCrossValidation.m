% Default, with acl
mapTrain;

trainParams.numReplicate = 20;
mapTrain;

trainParams.numReplicate = 25;
mapTrain;

trainParams.dropoutFraction = 0.7;
mapTrain;

trainParams.numReplicate = 20;
trainParams.dropoutFraction = 0.7;
mapTrain;

trainParams.numReplicate = 25;
trainParams.dropoutFraction = 0.7;
mapTrain;

% Repeat everything with turian.200
trainParams.wordDataset = 'turian.200';
mapTrain;
