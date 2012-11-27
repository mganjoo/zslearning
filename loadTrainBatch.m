function [ data, categories, categoryNames ] = loadTrainBatch( prefix, cifar_dir, batch )

if nargin < 3
    t = load([cifar_dir sprintf('/%s.mat', prefix)]);
else
    t = load([cifar_dir sprintf('/%s_%d.mat', prefix, batch)]);
end

data = t.trainX;
categories = t.trainY;
categoryNames = t.names;
clear t;

end
