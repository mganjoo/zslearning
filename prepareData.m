function [ data ] = prepareData( imgs, categories, wordTable )

numImages = size(imgs, 2);

data.imgs = imgs;
data.categories = categories;
data.wordTable = wordTable;

% prepare list of 'good' indices
data.goodIndices = (categories-1) * numImages + (1:numImages);

% set of good words
data.wgood = wordTable(:, categories);

end

