function [ data, categories, categoryNames ] = loadCIFAR10TrainBatch( prefix, batch, cifar_dir )

t = matfile([cifar_dir sprintf('/%s_%d.mat', prefix, batch)]);
data = t.trainX;
categories = t.trainY;
categoryNames = t.names;

end
