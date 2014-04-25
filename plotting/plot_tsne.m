% t-SNE visualize_script 
%WARNING: Needs lots of RAM !

% uncomment depending on what you want to map:
% training data (no zero shot classes) or test data (includes zero shot classes)
%  load('mappedTrainData.mat');
load('mappedTestData.mat');

load('word_data\acl\cifar10\wordTable.mat')

numImages = size(mappedX, 2);
t = tsne([mappedX wordTable]');
mappedX_t = t(1:numImages, :);
mappedWordTable_t = t(numImages+1:end, :);

% load label names
load('image_data/images/cifar10/meta.mat');

% load word table
load('word_data/acl/cifar10/wordTable.mat');

% do the visualization
visualize(mappedX_t, Y, mappedWordTable_t, label_names);
