function [ data, categories, categoryNames ] = loadTrainBatch( prefix, cifar_dir, batch )

if nargins < 3
    t = matfile([cifar_dir sprintf('/%s.mat', prefix)]);
else
    t = matfile([cifar_dir sprintf('/%s_%d.mat', prefix, batch)]);
end

data = t.trainX;
categories = t.trainY;
categoryNames = t.names;

end
