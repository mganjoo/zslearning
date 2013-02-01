function [mappedX, mappedWordTable] = map2D(mX, wordTable)
    addpath toolbox/;
    addpath tsne/;
    numImages = size(mX, 2);
    t = tsne([mX wordTable]');
    mappedX = t(1:numImages, :);
    mappedWordTable = t(numImages+1:end, :);
end