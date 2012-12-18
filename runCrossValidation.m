trainParams.imageDataset = 'cifar96';

wordDatasets = { 'acl', 'turian.200' };
dropoutFracs = [0.5 0.7];
numReplicates = [15 20 25];

for k = 1:length(wordDatasets)
    for i = 1:length(dropoutFracs)
        for j = 1:length(numReplicates)
            trainParams.wordDataset = wordDatasets(k);
            trainParams.dropoutFraction = dropoutFracs(i);
            trainParams.numReplicate = numReplicates(j);
            mapTrain;
        end
    end
end

clear;
